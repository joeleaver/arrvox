//! Procedural object snapshot for UI display.
//!
//! Plain data — no ECS, no hecs, no rinch. The engine builds this from
//! `ProceduralGeometry` and pushes it via `StateUpdate`. The editor reads
//! it to render the Build panel.

use glam::Vec3;

/// Snapshot of a procedural object's node tree for UI display.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ProceduralSnapshot {
    /// Entity UUID that owns this procedural object.
    pub entity_id: uuid::Uuid,
    /// Flat list of nodes (arena order).
    pub nodes: Vec<ProceduralNodeInfo>,
    /// Index of the root node.
    pub root: u32,
    /// Currently selected node (for param editing). None = no node selected.
    pub selected_node: Option<u32>,
}

/// Snapshot of a single node in the procedural tree.
#[derive(Debug, Clone, PartialEq)]
pub struct ProceduralNodeInfo {
    /// Arena index.
    pub id: u32,
    /// Display name (e.g., "Sphere", "Union", "Subtract").
    pub name: String,
    /// Short type label for the tree view icon.
    pub kind: ProceduralNodeKind,
    /// Child node IDs (arena indices).
    pub children: Vec<u32>,
    /// Whether this node is a leaf (can't have children).
    pub is_leaf: bool,
    /// Editable parameters for this node.
    pub params: Vec<ProceduralParam>,
}

/// Simplified node kind for UI display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProceduralNodeKind {
    Sphere,
    Box,
    Capsule,
    Cylinder,
    Torus,
    Plane,
    Union,
    Intersect,
    Subtract,
}

impl ProceduralNodeKind {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Sphere => "Sphere",
            Self::Box => "Box",
            Self::Capsule => "Capsule",
            Self::Cylinder => "Cylinder",
            Self::Torus => "Torus",
            Self::Plane => "Plane",
            Self::Union => "Union",
            Self::Intersect => "Intersect",
            Self::Subtract => "Subtract",
        }
    }
}

/// An editable parameter on a procedural node.
#[derive(Debug, Clone, PartialEq)]
pub struct ProceduralParam {
    pub name: String,
    pub value: ProceduralParamValue,
    pub range: Option<(f32, f32)>,
}

/// Parameter value types.
#[derive(Debug, Clone, PartialEq)]
pub enum ProceduralParamValue {
    Float(f32),
    Vec3([f32; 3]),
    U16(u16),
    MaterialCombine(String),
}

/// Build a `ProceduralSnapshot` from a `ProceduralGeometry` component.
pub fn build_procedural_snapshot(
    entity_id: uuid::Uuid,
    proc_geo: &crate::components::ProceduralGeometry,
    selected_node: Option<u32>,
) -> ProceduralSnapshot {
    use rkp_procedural::node_kind::*;
    use rkp_procedural::NodeKind;

    let tree = &proc_geo.tree;
    let mut nodes = Vec::new();

    for id in tree.iter_ids() {
        let node = tree.get(id).unwrap();
        let (kind, name, params) = match &node.kind {
            NodeKind::Sphere(p) => (
                ProceduralNodeKind::Sphere,
                "Sphere".to_string(),
                sphere_params(p),
            ),
            NodeKind::Box(p) => (
                ProceduralNodeKind::Box,
                "Box".to_string(),
                box_params(p),
            ),
            NodeKind::Capsule(p) => (
                ProceduralNodeKind::Capsule,
                "Capsule".to_string(),
                capsule_params(p),
            ),
            NodeKind::Cylinder(p) => (
                ProceduralNodeKind::Cylinder,
                "Cylinder".to_string(),
                cylinder_params(p),
            ),
            NodeKind::Torus(p) => (
                ProceduralNodeKind::Torus,
                "Torus".to_string(),
                torus_params(p),
            ),
            NodeKind::Plane(p) => (
                ProceduralNodeKind::Plane,
                "Plane".to_string(),
                plane_params(p),
            ),
            NodeKind::Union { material_combine } => (
                ProceduralNodeKind::Union,
                "Union".to_string(),
                combinator_params(material_combine),
            ),
            NodeKind::Intersect { material_combine } => (
                ProceduralNodeKind::Intersect,
                "Intersect".to_string(),
                combinator_params(material_combine),
            ),
            NodeKind::Subtract => (
                ProceduralNodeKind::Subtract,
                "Subtract".to_string(),
                vec![],
            ),
        };

        nodes.push(ProceduralNodeInfo {
            id: id.0,
            name,
            kind,
            children: node.children.iter().map(|c| c.0).collect(),
            is_leaf: node.kind.is_leaf(),
            params,
        });
    }

    ProceduralSnapshot {
        entity_id,
        nodes,
        root: tree.root().0,
        selected_node,
    }
}

fn sphere_params(p: &rkp_procedural::node_kind::SphereParams) -> Vec<ProceduralParam> {
    vec![
        ProceduralParam {
            name: "radius".into(),
            value: ProceduralParamValue::Float(p.radius),
            range: Some((0.01, 100.0)),
        },
        ProceduralParam {
            name: "falloff".into(),
            value: ProceduralParamValue::Float(p.falloff),
            range: Some((0.001, 10.0)),
        },
        ProceduralParam {
            name: "material_id".into(),
            value: ProceduralParamValue::U16(p.material_id),
            range: None,
        },
        ProceduralParam {
            name: "color".into(),
            value: ProceduralParamValue::Vec3(p.color.to_array()),
            range: None,
        },
    ]
}

fn box_params(p: &rkp_procedural::node_kind::BoxParams) -> Vec<ProceduralParam> {
    vec![
        ProceduralParam {
            name: "half_extents".into(),
            value: ProceduralParamValue::Vec3(p.half_extents.to_array()),
            range: None,
        },
        ProceduralParam {
            name: "rounding".into(),
            value: ProceduralParamValue::Float(p.rounding),
            range: Some((0.0, 10.0)),
        },
        ProceduralParam {
            name: "falloff".into(),
            value: ProceduralParamValue::Float(p.falloff),
            range: Some((0.001, 10.0)),
        },
        ProceduralParam {
            name: "material_id".into(),
            value: ProceduralParamValue::U16(p.material_id),
            range: None,
        },
        ProceduralParam {
            name: "color".into(),
            value: ProceduralParamValue::Vec3(p.color.to_array()),
            range: None,
        },
    ]
}

fn capsule_params(p: &rkp_procedural::node_kind::CapsuleParams) -> Vec<ProceduralParam> {
    vec![
        ProceduralParam {
            name: "half_height".into(),
            value: ProceduralParamValue::Float(p.half_height),
            range: Some((0.01, 100.0)),
        },
        ProceduralParam {
            name: "radius".into(),
            value: ProceduralParamValue::Float(p.radius),
            range: Some((0.01, 100.0)),
        },
        ProceduralParam {
            name: "falloff".into(),
            value: ProceduralParamValue::Float(p.falloff),
            range: Some((0.001, 10.0)),
        },
        ProceduralParam {
            name: "material_id".into(),
            value: ProceduralParamValue::U16(p.material_id),
            range: None,
        },
    ]
}

fn cylinder_params(p: &rkp_procedural::node_kind::CylinderParams) -> Vec<ProceduralParam> {
    vec![
        ProceduralParam {
            name: "half_height".into(),
            value: ProceduralParamValue::Float(p.half_height),
            range: Some((0.01, 100.0)),
        },
        ProceduralParam {
            name: "radius".into(),
            value: ProceduralParamValue::Float(p.radius),
            range: Some((0.01, 100.0)),
        },
        ProceduralParam {
            name: "falloff".into(),
            value: ProceduralParamValue::Float(p.falloff),
            range: Some((0.001, 10.0)),
        },
        ProceduralParam {
            name: "material_id".into(),
            value: ProceduralParamValue::U16(p.material_id),
            range: None,
        },
    ]
}

fn torus_params(p: &rkp_procedural::node_kind::TorusParams) -> Vec<ProceduralParam> {
    vec![
        ProceduralParam {
            name: "major_radius".into(),
            value: ProceduralParamValue::Float(p.major_radius),
            range: Some((0.01, 100.0)),
        },
        ProceduralParam {
            name: "minor_radius".into(),
            value: ProceduralParamValue::Float(p.minor_radius),
            range: Some((0.01, 50.0)),
        },
        ProceduralParam {
            name: "falloff".into(),
            value: ProceduralParamValue::Float(p.falloff),
            range: Some((0.001, 10.0)),
        },
        ProceduralParam {
            name: "material_id".into(),
            value: ProceduralParamValue::U16(p.material_id),
            range: None,
        },
    ]
}

fn plane_params(p: &rkp_procedural::node_kind::PlaneParams) -> Vec<ProceduralParam> {
    vec![
        ProceduralParam {
            name: "falloff".into(),
            value: ProceduralParamValue::Float(p.falloff),
            range: Some((0.001, 10.0)),
        },
        ProceduralParam {
            name: "material_id".into(),
            value: ProceduralParamValue::U16(p.material_id),
            range: None,
        },
    ]
}

fn combinator_params(mc: &rkp_procedural::MaterialCombine) -> Vec<ProceduralParam> {
    let value = match mc {
        rkp_procedural::MaterialCombine::Winner => "Winner",
        rkp_procedural::MaterialCombine::Layered => "Layered",
        rkp_procedural::MaterialCombine::Blend { .. } => "Blend",
    };
    vec![ProceduralParam {
        name: "material_combine".into(),
        value: ProceduralParamValue::MaterialCombine(value.to_string()),
        range: None,
    }]
}
