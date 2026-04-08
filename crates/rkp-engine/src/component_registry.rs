//! Component registry — runtime reflection for ECS components.
//!
//! Provides type-erased get_field/set_field operations so the inspector
//! can discover, read, and write ANY registered component's fields at runtime.

use crate::inspector::{FieldType, FieldValue};

/// Metadata for a single field on a component.
#[derive(Debug, Clone)]
pub struct FieldMeta {
    pub name: &'static str,
    pub field_type: FieldType,
    pub range: Option<(f64, f64)>,
    pub transient: bool,
    /// For FieldType::Struct — sub-field metadata.
    pub struct_fields: Option<&'static [FieldMeta]>,
    /// For FieldType::AssetRef — file extension filter (e.g., "rkp").
    pub asset_filter: Option<&'static str>,
}

/// Type-erased component operations.
///
/// Each registered component provides function pointers for:
/// - Checking if an entity has this component
/// - Reading a field by name (supports dot-notation for nested structs)
/// - Writing a field by name
/// - Adding a default instance to an entity
/// - Removing from an entity
pub struct ComponentEntry {
    pub name: &'static str,
    pub meta: &'static [FieldMeta],
    /// Can this component be removed from an entity? (Transform, EditorMetadata can't.)
    pub mandatory: bool,

    pub has: fn(&hecs::World, hecs::Entity) -> bool,
    pub get_field: fn(&hecs::World, hecs::Entity, &str) -> Result<FieldValue, String>,
    pub set_field: fn(&mut hecs::World, hecs::Entity, &str, FieldValue) -> Result<(), String>,
    pub add_default: fn(&mut hecs::World, hecs::Entity) -> Result<(), String>,
    pub remove: fn(&mut hecs::World, hecs::Entity) -> Result<(), String>,
}

/// Registry of all known component types.
pub struct ComponentRegistry {
    entries: Vec<ComponentEntry>,
}

impl ComponentRegistry {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Register a component type.
    pub fn register(&mut self, entry: ComponentEntry) {
        self.entries.push(entry);
    }

    /// Get a component entry by name.
    pub fn get(&self, name: &str) -> Option<&ComponentEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// List all registered component types.
    pub fn all(&self) -> &[ComponentEntry] {
        &self.entries
    }

    /// List components that are present on the given entity.
    pub fn components_on(&self, world: &hecs::World, entity: hecs::Entity) -> Vec<&ComponentEntry> {
        self.entries.iter().filter(|e| (e.has)(world, entity)).collect()
    }

    /// List components that are NOT present on the given entity (for "Add Component").
    pub fn available_for(&self, world: &hecs::World, entity: hecs::Entity) -> Vec<&ComponentEntry> {
        self.entries.iter().filter(|e| !(e.has)(world, entity)).collect()
    }
}

// ── Built-in component registrations ─────────────────────────────────

/// Register all built-in engine components.
pub fn register_builtins(registry: &mut ComponentRegistry) {
    registry.register(transform_entry());
    registry.register(editor_metadata_entry());
    registry.register(renderable_entry());
    registry.register(point_light_entry());
    registry.register(camera_entry());
}

// ── Transform ────────────────────────────────────────────────────────

static TRANSFORM_FIELDS: [FieldMeta; 3] = [
    FieldMeta { name: "position", field_type: FieldType::Vec3, range: None, transient: false, struct_fields: None, asset_filter: None },
    FieldMeta { name: "rotation", field_type: FieldType::Vec3, range: Some((-180.0, 180.0)), transient: false, struct_fields: None, asset_filter: None },
    FieldMeta { name: "scale", field_type: FieldType::Vec3, range: Some((0.01, 100.0)), transient: false, struct_fields: None, asset_filter: None },
];

fn transform_entry() -> ComponentEntry {
    use crate::components::Transform;
    ComponentEntry {
        name: "Transform",
        meta: &TRANSFORM_FIELDS,
        mandatory: true,
        has: |world, entity| world.get::<&Transform>(entity).is_ok(),
        get_field: |world, entity, field| {
            let c = world.get::<&Transform>(entity).map_err(|_| "no Transform".to_string())?;
            match field {
                "position" => Ok(FieldValue::Vec3(c.position.to_array())),
                "rotation" => Ok(FieldValue::Vec3(c.rotation.to_array())),
                "scale" => Ok(FieldValue::Vec3(c.scale.to_array())),
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        set_field: |world, entity, field, value| {
            let mut c = world.get::<&mut Transform>(entity).map_err(|_| "no Transform".to_string())?;
            match field {
                "position" => {
                    if let FieldValue::Vec3(v) = value { c.position = glam::Vec3::from_array(v); Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                "rotation" => {
                    if let FieldValue::Vec3(v) = value { c.rotation = glam::Vec3::from_array(v); Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                "scale" => {
                    if let FieldValue::Vec3(v) = value { c.scale = glam::Vec3::from_array(v); Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        add_default: |world, entity| {
            world.insert_one(entity, Transform::default()).map_err(|e| format!("{e}"))
        },
        remove: |_, _| Err("Transform is mandatory".into()),
    }
}

// ── EditorMetadata ───────────────────────────────────────────────────

static EDITOR_METADATA_FIELDS: [FieldMeta; 1] = [
    FieldMeta { name: "name", field_type: FieldType::String, range: None, transient: false, struct_fields: None, asset_filter: None },
];

fn editor_metadata_entry() -> ComponentEntry {
    use crate::components::EditorMetadata;
    ComponentEntry {
        name: "EditorMetadata",
        meta: &EDITOR_METADATA_FIELDS,
        mandatory: true,
        has: |world, entity| world.get::<&EditorMetadata>(entity).is_ok(),
        get_field: |world, entity, field| {
            let c = world.get::<&EditorMetadata>(entity).map_err(|_| "no EditorMetadata".to_string())?;
            match field {
                "name" => Ok(FieldValue::String(c.name.clone())),
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        set_field: |world, entity, field, value| {
            let mut c = world.get::<&mut EditorMetadata>(entity).map_err(|_| "no EditorMetadata".to_string())?;
            match field {
                "name" => {
                    if let FieldValue::String(v) = value { c.name = v; Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        add_default: |world, entity| {
            world.insert_one(entity, EditorMetadata::default()).map_err(|e| format!("{e}"))
        },
        remove: |_, _| Err("EditorMetadata is mandatory".into()),
    }
}

// ── Renderable ───────────────────────────────────────────────────────

static RENDERABLE_FIELDS: [FieldMeta; 4] = [
    FieldMeta { name: "asset_path", field_type: FieldType::String, range: None, transient: false, struct_fields: None, asset_filter: Some("rkp") },
    FieldMeta { name: "primitive", field_type: FieldType::String, range: None, transient: false, struct_fields: None, asset_filter: None },
    FieldMeta { name: "material_id", field_type: FieldType::Int, range: Some((0.0, 65535.0)), transient: false, struct_fields: None, asset_filter: None },
    FieldMeta { name: "voxel_count", field_type: FieldType::Int, range: None, transient: true, struct_fields: None, asset_filter: None },
];

fn renderable_entry() -> ComponentEntry {
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
    }
}

// ── PointLight ────────────────────────────────────────────────────────

static POINT_LIGHT_FIELDS: [FieldMeta; 3] = [
    FieldMeta { name: "color", field_type: FieldType::Color, range: None, transient: false, struct_fields: None, asset_filter: None },
    FieldMeta { name: "intensity", field_type: FieldType::Float, range: Some((0.0, 100.0)), transient: false, struct_fields: None, asset_filter: None },
    FieldMeta { name: "range", field_type: FieldType::Float, range: Some((0.1, 500.0)), transient: false, struct_fields: None, asset_filter: None },
];

fn point_light_entry() -> ComponentEntry {
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
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        add_default: |world, entity| {
            world.insert_one(entity, PointLight::default()).map_err(|e| format!("{e}"))
        },
        remove: |world, entity| {
            world.remove_one::<PointLight>(entity).map(|_| ()).map_err(|e| format!("{e}"))
        },
    }
}

// ── Camera ───────────────────────────────────────────────────────────

static CAMERA_FIELDS: [FieldMeta; 4] = [
    FieldMeta { name: "fov", field_type: FieldType::Float, range: Some((10.0, 170.0)), transient: false, struct_fields: None, asset_filter: None },
    FieldMeta { name: "near", field_type: FieldType::Float, range: Some((0.001, 10.0)), transient: false, struct_fields: None, asset_filter: None },
    FieldMeta { name: "far", field_type: FieldType::Float, range: Some((10.0, 100000.0)), transient: false, struct_fields: None, asset_filter: None },
    FieldMeta { name: "active", field_type: FieldType::Bool, range: None, transient: false, struct_fields: None, asset_filter: None },
];

fn camera_entry() -> ComponentEntry {
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
    }
}
