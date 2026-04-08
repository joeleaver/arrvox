//! Inspector data model — snapshots of entity components for UI display/editing.
//!
//! The engine builds `InspectorSnapshot` from ECS state and pushes it to the UI.
//! The UI renders type-specific editors for each field. Edits go back via
//! `EngineCommand::SetComponentField`.

use serde::{Deserialize, Serialize};

/// Dynamically-typed value for component fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FieldValue {
    Float(f64),
    Int(i64),
    Bool(bool),
    String(String),
    Vec3([f32; 3]),
    Color([f32; 4]),
}

impl Default for FieldValue {
    fn default() -> Self {
        Self::Float(0.0)
    }
}

impl std::fmt::Display for FieldValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Float(v) => write!(f, "{v:.3}"),
            Self::Int(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::String(v) => write!(f, "{v}"),
            Self::Vec3(v) => write!(f, "[{:.2}, {:.2}, {:.2}]", v[0], v[1], v[2]),
            Self::Color(v) => write!(f, "[{:.2}, {:.2}, {:.2}, {:.2}]", v[0], v[1], v[2], v[3]),
        }
    }
}

/// What kind of editor to render for a field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FieldType {
    #[default]
    Float,
    Int,
    Bool,
    String,
    Vec3,
    Color,
}

/// Metadata about a single field in a component.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FieldSnapshot {
    pub name: String,
    pub field_type: FieldType,
    pub value: FieldValue,
    /// Optional range for numeric fields (min, max).
    pub range: Option<(f64, f64)>,
}

/// Snapshot of a single component on an entity.
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentSnapshot {
    pub name: String,
    pub fields: Vec<FieldSnapshot>,
    /// Whether this component can be removed (mandatory components like Transform can't).
    pub removable: bool,
}

/// Full inspector snapshot for a selected entity.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct InspectorSnapshot {
    pub entity_name: String,
    pub entity_id: String,
    pub position: [f32; 3],
    pub rotation: [f32; 3],
    pub scale: [f32; 3],
    pub components: Vec<ComponentSnapshot>,
}
