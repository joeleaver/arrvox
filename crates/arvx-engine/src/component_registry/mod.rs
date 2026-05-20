//! Component registry — runtime reflection for ECS components.
//!
//! Provides type-erased get_field/set_field operations so the inspector
//! can discover, read, and write ANY registered component's fields at runtime.

use crate::inspector::{FieldType, FieldValue};

mod sim;
mod terrain_stamp;
mod visuals;
mod world;

/// Metadata for a single field on a component.
#[derive(Debug, Clone)]
pub struct FieldMeta {
    pub name: &'static str,
    pub field_type: FieldType,
    pub range: Option<(f64, f64)>,
    pub transient: bool,
    /// For FieldType::Struct — sub-field metadata.
    pub struct_fields: Option<&'static [FieldMeta]>,
    /// For FieldType::AssetRef — file extension filter (e.g., "arvx").
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
/// `#[arvx_component]` proc macro.
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
    registry.register(world::transform_entry());
    registry.register(world::editor_metadata_entry());
    registry.register(visuals::renderable_entry());
    registry.register(visuals::point_light_entry());
    registry.register(visuals::camera_entry());
    registry.register(visuals::spot_light_entry());
    registry.register(sim::rigid_body_entry());
    registry.register(world::procedural_geometry_entry());
    registry.register(sim::skeleton_entry());
    registry.register(sim::animation_player_entry());
    registry.register(crate::generator::generator_state_entry());
    registry.register(crate::generator::generator_owned_entry());
    registry.register(terrain_stamp::stamp_entry());
}

