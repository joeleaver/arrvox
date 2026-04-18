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
    /// Transient fields are runtime-only (not persisted).
    pub transient: bool,
    /// For AssetRef fields — file extension filter.
    pub asset_filter: Option<String>,
    /// For enum-like String fields — valid (value, label) pairs for a dropdown.
    pub enum_options: Vec<(String, String)>,
    /// Use scrub input instead of slider for ranged floats.
    pub scrub: bool,
}

/// Snapshot of a single component on an entity.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ComponentSnapshot {
    pub name: String,
    pub fields: Vec<FieldSnapshot>,
    /// Whether this component can be removed (mandatory components like Transform can't).
    pub removable: bool,
}

/// Per-material voxel usage for a renderable entity.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MaterialUsage {
    /// Material runtime ID.
    pub material_id: u16,
    /// How many voxels use this as their primary material.
    pub voxel_count: u32,
}

/// Skeletal-animation sidecar on an inspector snapshot. Populated only
/// when the selected entity has a `Skeleton` component; carries the
/// per-asset data the UI needs to render a clip picker and bone tree
/// (both require knowledge the static component reflection can't
/// supply — clips and bones are asset data, not component fields).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SkeletonInspector {
    /// Absolute path of the loaded `.rkskel`.
    pub path: String,
    /// Bone names in index order.
    pub bone_names: Vec<String>,
    /// Parent index per bone. `-1` means root. Parallel to `bone_names`.
    pub bone_parents: Vec<i32>,
    /// Animation clips available on this skeleton asset.
    pub clips: Vec<ClipInfo>,
}

/// One animation clip bundled with a skeleton asset.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ClipInfo {
    pub name: String,
    pub duration: f32,
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
    /// Per-material voxel usage (only for entities with Renderable + spatial data).
    pub material_usage: Vec<MaterialUsage>,
    /// Set when the entity has a loaded skeleton.
    pub skeleton: Option<SkeletonInspector>,
}
