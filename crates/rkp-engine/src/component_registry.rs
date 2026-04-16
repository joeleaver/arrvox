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
    /// For enum-like String fields — list of valid values for a dropdown.
    /// Each entry is (value, display_label). Empty = free-form text.
    pub enum_options: Option<&'static [(&'static str, &'static str)]>,
    /// Use a scrub input (drag-to-change number) instead of a slider.
    pub scrub: bool,
}

/// Type-erased component operations.
///
/// Each registered component provides function pointers for:
/// - Checking if an entity has this component
/// - Reading/writing a field by name
/// - Adding a default instance / removing from an entity
/// - Serializing to / deserializing from JSON
///
/// Components are auto-registered via `inventory::submit!` from the
/// `#[rkp_component]` proc macro.
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
    /// Serialize component data to JSON. Returns None if entity doesn't have this component.
    pub serialize: fn(&hecs::World, hecs::Entity) -> Option<String>,
    /// Deserialize JSON and insert onto entity.
    pub deserialize_insert: fn(&mut hecs::World, hecs::Entity, &str) -> Result<(), String>,

    /// Called when this component is added to an entity (during command flush).
    pub on_add: Option<fn(&mut hecs::World, hecs::Entity)>,
    /// Called when this component is about to be removed from an entity (during command flush).
    pub on_remove: Option<fn(&mut hecs::World, hecs::Entity)>,
}

inventory::collect!(ComponentEntry);

/// Registry of all known component types.
pub struct ComponentRegistry {
    /// Auto-discovered via inventory (same binary).
    entries: Vec<&'static ComponentEntry>,
    /// Manually registered built-in components.
    manual_entries: Vec<ComponentEntry>,
    /// Gameplay components from hot-reloaded dylib.
    gameplay_entries: Vec<&'static ComponentEntry>,
}

impl ComponentRegistry {
    /// Create and populate from inventory (auto-registered components).
    pub fn new() -> Self {
        let entries: Vec<&'static ComponentEntry> = inventory::iter::<ComponentEntry>.into_iter().collect();
        Self { entries, manual_entries: Vec::new(), gameplay_entries: Vec::new() }
    }

    /// Manually register a component (for built-in components not using the macro).
    pub fn register(&mut self, entry: ComponentEntry) {
        self.manual_entries.push(entry);
    }

    /// Register a gameplay component entry (from hot-reloaded dylib).
    pub fn register_gameplay(&mut self, entry: &'static ComponentEntry) {
        // Avoid duplicates.
        if !self.gameplay_entries.iter().any(|e| e.name == entry.name) {
            self.gameplay_entries.push(entry);
        }
    }

    /// Remove all gameplay component entries (before dylib unload).
    pub fn clear_gameplay(&mut self) {
        self.gameplay_entries.clear();
    }

    /// Get a component entry by name.
    pub fn get(&self, name: &str) -> Option<&ComponentEntry> {
        self.entries.iter().find(|e| e.name == name).map(|e| *e)
            .or_else(|| self.manual_entries.iter().find(|e| e.name == name))
            .or_else(|| self.gameplay_entries.iter().find(|e| e.name == name).map(|e| *e))
    }

    /// Iterate all registered component entries.
    fn all_entries(&self) -> impl Iterator<Item = &ComponentEntry> {
        self.entries.iter().map(|e| *e)
            .chain(self.manual_entries.iter())
            .chain(self.gameplay_entries.iter().map(|e| *e))
    }

    /// List components that are present on the given entity.
    pub fn components_on(&self, world: &hecs::World, entity: hecs::Entity) -> Vec<&ComponentEntry> {
        self.all_entries().filter(|e| (e.has)(world, entity)).collect()
    }

    /// List components that are NOT present on the given entity (for "Add Component").
    pub fn available_for(&self, world: &hecs::World, entity: hecs::Entity) -> Vec<&ComponentEntry> {
        self.all_entries().filter(|e| !(e.has)(world, entity)).collect()
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
    registry.register(spot_light_entry());
    registry.register(rigid_body_entry());
    registry.register(procedural_geometry_entry());
}

// ── ProceduralGeometry ───────────────────────────────────────────────

/// Small surface: the tree itself is edited via the build panel, not the
/// inspector. We expose only the two scalars that make sense to tweak
/// without going through the build panel (voxel size tier, collider
/// resolution). Registration exists primarily so `ProceduralGeometry`
/// participates in `.rkproject` save / load — without it, procedurals
/// silently lose their tree on reopen.
static PROCEDURAL_FIELDS: [FieldMeta; 2] = [
    FieldMeta { name: "voxel_size", field_type: FieldType::Float, range: Some((0.005, 0.32)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: true },
    FieldMeta { name: "collider_resolution", field_type: FieldType::Float, range: Some((0.05, 1.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: true },
];

fn procedural_geometry_entry() -> ComponentEntry {
    use crate::components::ProceduralGeometry;
    ComponentEntry {
        name: "ProceduralGeometry",
        meta: &PROCEDURAL_FIELDS,
        mandatory: false,
        has: |world, entity| world.get::<&ProceduralGeometry>(entity).is_ok(),
        get_field: |world, entity, field| {
            let c = world.get::<&ProceduralGeometry>(entity).map_err(|_| "no ProceduralGeometry".to_string())?;
            match field {
                "voxel_size" => Ok(FieldValue::Float(c.voxel_size as f64)),
                "collider_resolution" => Ok(FieldValue::Float(c.collider_resolution as f64)),
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        set_field: |world, entity, field, value| {
            let mut c = world.get::<&mut ProceduralGeometry>(entity).map_err(|_| "no ProceduralGeometry".to_string())?;
            match field {
                "voxel_size" => {
                    if let FieldValue::Float(v) = value { c.voxel_size = v as f32; c.dirty = true; Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                "collider_resolution" => {
                    if let FieldValue::Float(v) = value { c.collider_resolution = v as f32; Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        add_default: |world, entity| {
            world.insert_one(entity, ProceduralGeometry::default_sphere()).map_err(|e| format!("{e}"))
        },
        remove: |world, entity| {
            world.remove_one::<ProceduralGeometry>(entity).map(|_| ()).map_err(|e| format!("{e}"))
        },
        serialize: |world, entity| {
            let c = world.get::<&ProceduralGeometry>(entity).ok()?;
            serde_json::to_string(&*c).ok()
        },
        deserialize_insert: |world, entity, json| {
            let c: ProceduralGeometry = serde_json::from_str(json).map_err(|e| format!("{e}"))?;
            world.insert_one(entity, c).map_err(|e| format!("{e}"))
        },
        on_add: None,
        on_remove: None,
    }
}

// ── Transform ────────────────────────────────────────────────────────

static TRANSFORM_FIELDS: [FieldMeta; 3] = [
    FieldMeta { name: "position", field_type: FieldType::Vec3, range: None, transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
    FieldMeta { name: "rotation", field_type: FieldType::Vec3, range: Some((-180.0, 180.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
    FieldMeta { name: "scale", field_type: FieldType::Vec3, range: Some((0.01, 100.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
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
        serialize: |world, entity| {
            let c = world.get::<&Transform>(entity).ok()?;
            serde_json::to_string(&*c).ok()
        },
        deserialize_insert: |world, entity, json| {
            let c: Transform = serde_json::from_str(json).map_err(|e| format!("{e}"))?;
            world.insert_one(entity, c).map_err(|e| format!("{e}"))
        },
        on_add: None,
        on_remove: None,
    }
}

// ── EditorMetadata ───────────────────────────────────────────────────

static EDITOR_METADATA_FIELDS: [FieldMeta; 1] = [
    FieldMeta { name: "name", field_type: FieldType::String, range: None, transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
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
        serialize: |world, entity| {
            let c = world.get::<&EditorMetadata>(entity).ok()?;
            serde_json::to_string(&*c).ok()
        },
        deserialize_insert: |world, entity, json| {
            let c: EditorMetadata = serde_json::from_str(json).map_err(|e| format!("{e}"))?;
            world.insert_one(entity, c).map_err(|e| format!("{e}"))
        },
        on_add: None,
        on_remove: None,
    }
}

// ── Renderable ───────────────────────────────────────────────────────

static RENDERABLE_FIELDS: [FieldMeta; 4] = [
    FieldMeta { name: "asset_path", field_type: FieldType::String, range: None, transient: false, struct_fields: None, asset_filter: Some("rkp"), enum_options: None, scrub: false },
    FieldMeta { name: "primitive", field_type: FieldType::String, range: None, transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
    FieldMeta { name: "material_id", field_type: FieldType::Int, range: Some((0.0, 65535.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
    FieldMeta { name: "voxel_count", field_type: FieldType::Int, range: None, transient: true, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
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
    }
}

// ── PointLight ────────────────────────────────────────────────────────

static POINT_LIGHT_FIELDS: [FieldMeta; 4] = [
    FieldMeta { name: "color", field_type: FieldType::Color, range: None, transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
    FieldMeta { name: "intensity", field_type: FieldType::Float, range: Some((0.0, 100_000.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: true },
    FieldMeta { name: "range", field_type: FieldType::Float, range: Some((0.1, 500.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: true },
    FieldMeta { name: "cast_shadow", field_type: FieldType::Bool, range: None, transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
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

fn spot_light_entry() -> ComponentEntry {
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
    }
}

// ── Camera ───────────────────────────────────────────────────────────

static CAMERA_FIELDS: [FieldMeta; 4] = [
    FieldMeta { name: "fov", field_type: FieldType::Float, range: Some((10.0, 170.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
    FieldMeta { name: "near", field_type: FieldType::Float, range: Some((0.001, 10.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
    FieldMeta { name: "far", field_type: FieldType::Float, range: Some((10.0, 100000.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
    FieldMeta { name: "active", field_type: FieldType::Bool, range: None, transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
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
    }
}

// ── RigidBody ───────────────────────────────────────────────────────

static RIGID_BODY_FIELDS: [FieldMeta; 6] = [
    FieldMeta { name: "body_type", field_type: FieldType::String, range: None, transient: false, struct_fields: None, asset_filter: None,
        enum_options: Some(&[("Dynamic", "Dynamic"), ("Static", "Static"), ("KinematicPosition", "Kinematic Pos"), ("KinematicVelocity", "Kinematic Vel")]), scrub: false },
    FieldMeta { name: "collider_shape", field_type: FieldType::String, range: None, transient: false, struct_fields: None, asset_filter: None,
        enum_options: Some(&[("Auto", "Auto (Voxel)"), ("Box", "Box"), ("Sphere", "Sphere"), ("Capsule", "Capsule")]), scrub: false },
    FieldMeta { name: "mass", field_type: FieldType::Float, range: Some((0.01, 1000.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
    FieldMeta { name: "friction", field_type: FieldType::Float, range: Some((0.0, 2.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
    FieldMeta { name: "restitution", field_type: FieldType::Float, range: Some((0.0, 1.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
    FieldMeta { name: "collider_cell_size", field_type: FieldType::Float, range: Some((0.01, 1.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: true },
];

fn rigid_body_entry() -> ComponentEntry {
    use crate::components::RigidBody;
    ComponentEntry {
        name: "RigidBody",
        meta: &RIGID_BODY_FIELDS,
        mandatory: false,
        has: |world, entity| world.get::<&RigidBody>(entity).is_ok(),
        get_field: |world, entity, field| {
            let c = world.get::<&RigidBody>(entity).map_err(|_| "no RigidBody".to_string())?;
            match field {
                "body_type" => Ok(FieldValue::String(format!("{:?}", c.body_type))),
                "collider_shape" => Ok(FieldValue::String(format!("{:?}", c.collider_shape))),
                "mass" => Ok(FieldValue::Float(c.mass as f64)),
                "friction" => Ok(FieldValue::Float(c.friction as f64)),
                "restitution" => Ok(FieldValue::Float(c.restitution as f64)),
                "collider_cell_size" => Ok(FieldValue::Float(c.collider_cell_size as f64)),
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        set_field: |world, entity, field, value| {
            let mut c = world.get::<&mut RigidBody>(entity).map_err(|_| "no RigidBody".to_string())?;
            match field {
                "body_type" => {
                    if let FieldValue::String(v) = value {
                        c.body_type = match v.as_str() {
                            "Static" => rkf_physics::rigid_body::BodyType::Static,
                            "KinematicPosition" => rkf_physics::rigid_body::BodyType::KinematicPosition,
                            "KinematicVelocity" => rkf_physics::rigid_body::BodyType::KinematicVelocity,
                            _ => rkf_physics::rigid_body::BodyType::Dynamic,
                        };
                        Ok(())
                    } else { Err("type mismatch".into()) }
                }
                "collider_shape" => {
                    if let FieldValue::String(v) = value {
                        c.collider_shape = match v.as_str() {
                            "Box" => rkf_physics::rigid_body::ColliderShape::Box,
                            "Sphere" => rkf_physics::rigid_body::ColliderShape::Sphere,
                            "Capsule" => rkf_physics::rigid_body::ColliderShape::Capsule,
                            _ => rkf_physics::rigid_body::ColliderShape::Auto,
                        };
                        Ok(())
                    } else { Err("type mismatch".into()) }
                }
                "mass" => {
                    if let FieldValue::Float(v) = value { c.mass = v as f32; Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                "friction" => {
                    if let FieldValue::Float(v) = value { c.friction = v as f32; Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                "restitution" => {
                    if let FieldValue::Float(v) = value { c.restitution = v as f32; Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                "collider_cell_size" => {
                    if let FieldValue::Float(v) = value { c.collider_cell_size = v as f32; Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        add_default: |world, entity| {
            world.insert_one(entity, RigidBody::default()).map_err(|e| format!("{e}"))
        },
        remove: |world, entity| {
            world.remove_one::<RigidBody>(entity).map(|_| ()).map_err(|e| format!("{e}"))
        },
        serialize: |world, entity| {
            let c = world.get::<&RigidBody>(entity).ok()?;
            serde_json::to_string(&*c).ok()
        },
        deserialize_insert: |world, entity, json| {
            let c: RigidBody = serde_json::from_str(json).map_err(|e| format!("{e}"))?;
            world.insert_one(entity, c).map_err(|e| format!("{e}"))
        },
        on_add: None,
        on_remove: None,
    }
}
