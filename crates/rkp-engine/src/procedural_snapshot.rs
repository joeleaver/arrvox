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
    /// Render voxel size.
    pub voxel_size: f32,
    /// Whether the bake worker should voxelize this procedural
    /// (default) or emit a triangle proxy mesh.
    pub bake_mode: crate::components::BakeMode,
    /// Tree has been edited since the last voxel bake — the "Bake" button
    /// should show as enabled/highlighted when this is true. Includes
    /// both `dirty` (build-panel param edits awaiting an explicit
    /// bake) and `pending_bake` (auto-bake debouncing or in flight from
    /// the properties-panel scale slider) so any UI surface that shows
    /// "baked vs unbaked" gets the same answer.
    pub dirty: bool,
    /// Voxel count from the last successful bake (zero pre-bake). Read
    /// from the entity's `Renderable` component so the build overlay
    /// can show a live count next to the Bake button without a second
    /// snapshot hop.
    pub voxel_count: u32,
}

/// Snapshot of a single node in the procedural tree.
#[derive(Debug, Clone, PartialEq, Default)]
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
    /// Whether this is the root node.
    pub is_root: bool,
    /// Local translation (from node transform).
    pub position: [f32; 3],
    /// Local rotation, Euler degrees (XYZ order) — decomposed from the
    /// node's `Affine3A`. Matches the entity Transform convention.
    pub rotation: [f32; 3],
    /// Local scale factor per axis.
    pub scale: [f32; 3],
    /// Editable parameters for this node.
    pub params: Vec<ProceduralParam>,
    /// Whether the "+" add-child affordance should be visible on this
    /// row. True for combinators under their child cap (unbounded for
    /// Union/Intersect/Subtract; 1 for single-child effects like
    /// NoiseDisplace). Leaves default to false — the engine already
    /// auto-promotes a leaf root to a Union on the first add, so the
    /// root-leaf case is handled with a separate `is_root` branch in
    /// the UI rather than this flag.
    pub can_add_child: bool,
}

/// Simplified node kind for UI display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProceduralNodeKind {
    #[default]
    Root,
    Sphere,
    Box,
    Capsule,
    Cylinder,
    Torus,
    Plane,
    Ramp,
    Union,
    Intersect,
    Subtract,
    NoiseDisplace,
    Mirror,
    MaterialByHeight,
    ColorByHeight,
    MaterialByNoise,
    ColorByNoise,
    Array,
}

impl ProceduralNodeKind {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Root => "Root",
            Self::Sphere => "Sphere",
            Self::Box => "Box",
            Self::Capsule => "Capsule",
            Self::Cylinder => "Cylinder",
            Self::Torus => "Torus",
            Self::Plane => "Plane",
            Self::Ramp => "Ramp",
            Self::Union => "Union",
            Self::Intersect => "Intersect",
            Self::Subtract => "Subtract",
            Self::NoiseDisplace => "Noise Displace",
            Self::Mirror => "Mirror",
            Self::MaterialByHeight => "Material by Height",
            Self::ColorByHeight => "Color by Height",
            Self::MaterialByNoise => "Material by Noise",
            Self::ColorByNoise => "Color by Noise",
            Self::Array => "Array",
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
    /// RGBA color. Alpha is tracked for the color-picker control even
    /// though leaf params today only store RGB — the picker round-trips
    /// through hex strings that would otherwise require re-emitting an
    /// alpha channel anyway.
    Color([f32; 4]),
    /// Material-palette reference. Rendered as a drag-drop slot (swatch +
    /// name) that accepts materials from the materials panel.
    Material(u16),
    MaterialCombine(String),
    /// Generic string-enumerated picker. `value` is the current choice;
    /// `options` is a list of `(value, label)` pairs the UI renders as a
    /// dropdown. Added for `Mirror`'s `axis` param but intended to serve
    /// any future effect that needs a small fixed set of choices.
    Select {
        value: String,
        options: Vec<(String, String)>,
    },
}

/// Build a `ProceduralSnapshot` from a `ProceduralGeometry` component.
pub fn build_procedural_snapshot(
    entity_id: uuid::Uuid,
    proc_geo: &crate::components::ProceduralGeometry,
    selected_node: Option<u32>,
    voxel_size: f32,
    voxel_count: u32,
) -> ProceduralSnapshot {
    use rkp_procedural::NodeKind;

    let tree = &proc_geo.tree;
    let mut nodes = Vec::new();

    for id in tree.iter_ids() {
        let node = tree.get(id).unwrap();
        let (kind, name, params) = match &node.kind {
            NodeKind::Root => (
                ProceduralNodeKind::Root,
                "Root".to_string(),
                vec![],
            ),
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
            NodeKind::Ramp(p) => (
                ProceduralNodeKind::Ramp,
                "Ramp".to_string(),
                ramp_params(p),
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
            NodeKind::NoiseDisplace(p) => (
                ProceduralNodeKind::NoiseDisplace,
                "Noise Displace".to_string(),
                noise_displace_params(p),
            ),
            NodeKind::Mirror(p) => (
                ProceduralNodeKind::Mirror,
                "Mirror".to_string(),
                mirror_params(p),
            ),
            NodeKind::MaterialByHeight(p) => (
                ProceduralNodeKind::MaterialByHeight,
                "Material by Height".to_string(),
                material_by_height_params(p),
            ),
            NodeKind::ColorByHeight(p) => (
                ProceduralNodeKind::ColorByHeight,
                "Color by Height".to_string(),
                color_by_height_params(p),
            ),
            NodeKind::MaterialByNoise(p) => (
                ProceduralNodeKind::MaterialByNoise,
                "Material by Noise".to_string(),
                material_by_noise_params(p),
            ),
            NodeKind::ColorByNoise(p) => (
                ProceduralNodeKind::ColorByNoise,
                "Color by Noise".to_string(),
                color_by_noise_params(p),
            ),
            NodeKind::Array(p) => (
                ProceduralNodeKind::Array,
                "Array".to_string(),
                array_params(p),
            ),
        };

        let (position, rotation_deg, scale) = decompose_affine(&node.transform);
        // "+" is shown when the kind is a combinator-style container
        // AND the child count is below its cap. Leaves have `Some(0)`
        // so `can_add_child` is false; unbounded kinds have `None` and
        // always return true.
        let child_count = node.children.len();
        let at_cap = node
            .kind
            .max_children()
            .is_some_and(|cap| child_count >= cap);
        let can_add_child = !node.kind.is_leaf() && !at_cap;

        nodes.push(ProceduralNodeInfo {
            id: id.0,
            name,
            kind,
            children: node.children.iter().map(|c| c.0).collect(),
            is_leaf: node.kind.is_leaf(),
            is_root: id == tree.root(),
            position,
            rotation: rotation_deg,
            scale,
            params,
            can_add_child,
        });
    }

    ProceduralSnapshot {
        entity_id,
        nodes,
        root: tree.root().0,
        selected_node,
        voxel_size,
        bake_mode: proc_geo.bake_mode,
        dirty: proc_geo.dirty || proc_geo.pending_bake || proc_geo.bake_in_flight,
        voxel_count,
    }
}

/// Split a node's local `Affine3A` into translation, Euler rotation
/// (degrees, XYZ order), and per-axis scale.
///
/// The node transforms stored on the tree are always rigid-uniform +
/// non-uniform scale composed by the builder (`from_scale_rotation_
/// translation`). Decomposition therefore does NOT need to handle
/// shear — we extract per-axis scale as column lengths of the upper
/// 3×3, strip it out to get the pure rotation matrix, then convert to
/// a quaternion and into XYZ Euler degrees.
fn decompose_affine(t: &glam::Affine3A) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let translation = t.translation;
    let m = t.matrix3;
    let sx = Vec3::from(m.x_axis).length();
    let sy = Vec3::from(m.y_axis).length();
    let sz = Vec3::from(m.z_axis).length();
    let safe = |v: f32| if v.abs() < 1e-8 { 1.0 } else { v };
    let rot_mat = glam::Mat3::from_cols(
        (Vec3::from(m.x_axis) / safe(sx)).into(),
        (Vec3::from(m.y_axis) / safe(sy)).into(),
        (Vec3::from(m.z_axis) / safe(sz)).into(),
    );
    let quat = glam::Quat::from_mat3(&rot_mat);
    let (x, y, z) = quat.to_euler(glam::EulerRot::XYZ);
    (
        [translation.x, translation.y, translation.z],
        [x.to_degrees(), y.to_degrees(), z.to_degrees()],
        [sx, sy, sz],
    )
}

// Dimension-like params (radius, half_extents, half_height, ...) are
// intentionally omitted from the UI: the Scale transform on the node
// covers the same space (and gives non-uniform scaling / ellipsoids /
// scalene cylinders for free). The defaults are canonical unit shapes —
// sphere r=0.5, unit cube, unit-ish capsule/cylinder/ramp — so the
// transform's scale factor reads as "size in world units."
//
// Torus is the one exception: `minor_radius` (tube thickness) is
// independent of the ring radius and can't be expressed through a
// Vec3 scale, so it stays visible.

fn rgba_from_color(c: glam::Vec3) -> [f32; 4] {
    [c.x, c.y, c.z, 1.0]
}

fn sphere_params(p: &rkp_procedural::node_kind::SphereParams) -> Vec<ProceduralParam> {
    vec![
        ProceduralParam { name: "material".into(), value: ProceduralParamValue::Material(p.material_id), range: None },
        ProceduralParam { name: "color".into(), value: ProceduralParamValue::Color(rgba_from_color(p.color)), range: None },
    ]
}

fn box_params(p: &rkp_procedural::node_kind::BoxParams) -> Vec<ProceduralParam> {
    vec![
        ProceduralParam { name: "rounding".into(), value: ProceduralParamValue::Float(p.rounding), range: Some((0.0, 10.0)) },
        ProceduralParam { name: "material".into(), value: ProceduralParamValue::Material(p.material_id), range: None },
        ProceduralParam { name: "color".into(), value: ProceduralParamValue::Color(rgba_from_color(p.color)), range: None },
    ]
}

fn capsule_params(p: &rkp_procedural::node_kind::CapsuleParams) -> Vec<ProceduralParam> {
    vec![
        ProceduralParam { name: "material".into(), value: ProceduralParamValue::Material(p.material_id), range: None },
    ]
}

fn cylinder_params(p: &rkp_procedural::node_kind::CylinderParams) -> Vec<ProceduralParam> {
    vec![
        ProceduralParam { name: "material".into(), value: ProceduralParamValue::Material(p.material_id), range: None },
    ]
}

fn torus_params(p: &rkp_procedural::node_kind::TorusParams) -> Vec<ProceduralParam> {
    vec![
        // Exposed as `tube_radius` in the UI — the TorusParams field is
        // still `minor_radius` underneath. Setter accepts both names.
        ProceduralParam { name: "tube_radius".into(), value: ProceduralParamValue::Float(p.minor_radius), range: Some((0.01, 50.0)) },
        ProceduralParam { name: "material".into(), value: ProceduralParamValue::Material(p.material_id), range: None },
    ]
}

fn plane_params(p: &rkp_procedural::node_kind::PlaneParams) -> Vec<ProceduralParam> {
    vec![
        ProceduralParam { name: "material".into(), value: ProceduralParamValue::Material(p.material_id), range: None },
    ]
}

fn ramp_params(p: &rkp_procedural::node_kind::RampParams) -> Vec<ProceduralParam> {
    vec![
        ProceduralParam { name: "material".into(), value: ProceduralParamValue::Material(p.material_id), range: None },
        ProceduralParam { name: "color".into(), value: ProceduralParamValue::Color(rgba_from_color(p.color)), range: None },
    ]
}

fn noise_displace_params(
    p: &rkp_procedural::node_kind::NoiseDisplaceParams,
) -> Vec<ProceduralParam> {
    vec![
        ProceduralParam { name: "amplitude".into(), value: ProceduralParamValue::Float(p.amplitude), range: Some((0.0, 2.0)) },
        ProceduralParam { name: "frequency".into(), value: ProceduralParamValue::Float(p.frequency), range: Some((0.05, 32.0)) },
        ProceduralParam { name: "octaves".into(), value: ProceduralParamValue::Float(p.octaves as f32), range: Some((1.0, 8.0)) },
        ProceduralParam { name: "seed".into(), value: ProceduralParamValue::Float(p.seed as f32), range: Some((0.0, 1024.0)) },
    ]
}

fn material_by_height_params(
    p: &rkp_procedural::node_kind::MaterialByHeightParams,
) -> Vec<ProceduralParam> {
    vec![
        ProceduralParam {
            name: "low_material".into(),
            value: ProceduralParamValue::Material(p.low_material),
            range: None,
        },
        ProceduralParam {
            name: "low_to_mid".into(),
            value: ProceduralParamValue::Float(p.low_to_mid),
            range: Some((-100.0, 100.0)),
        },
        ProceduralParam {
            name: "mid_material".into(),
            value: ProceduralParamValue::Material(p.mid_material),
            range: None,
        },
        ProceduralParam {
            name: "mid_to_high".into(),
            value: ProceduralParamValue::Float(p.mid_to_high),
            range: Some((-100.0, 100.0)),
        },
        ProceduralParam {
            name: "high_material".into(),
            value: ProceduralParamValue::Material(p.high_material),
            range: None,
        },
        ProceduralParam {
            name: "transition_width".into(),
            value: ProceduralParamValue::Float(p.transition_width),
            range: Some((0.0, 10.0)),
        },
    ]
}

fn color_by_height_params(
    p: &rkp_procedural::node_kind::ColorByHeightParams,
) -> Vec<ProceduralParam> {
    vec![
        ProceduralParam {
            name: "low_color".into(),
            value: ProceduralParamValue::Color(rgba_from_color(p.low_color)),
            range: None,
        },
        ProceduralParam {
            name: "low_to_mid".into(),
            value: ProceduralParamValue::Float(p.low_to_mid),
            range: Some((-100.0, 100.0)),
        },
        ProceduralParam {
            name: "mid_color".into(),
            value: ProceduralParamValue::Color(rgba_from_color(p.mid_color)),
            range: None,
        },
        ProceduralParam {
            name: "mid_to_high".into(),
            value: ProceduralParamValue::Float(p.mid_to_high),
            range: Some((-100.0, 100.0)),
        },
        ProceduralParam {
            name: "high_color".into(),
            value: ProceduralParamValue::Color(rgba_from_color(p.high_color)),
            range: None,
        },
        ProceduralParam {
            name: "transition_width".into(),
            value: ProceduralParamValue::Float(p.transition_width),
            range: Some((0.0, 10.0)),
        },
    ]
}

fn material_by_noise_params(
    p: &rkp_procedural::node_kind::MaterialByNoiseParams,
) -> Vec<ProceduralParam> {
    vec![
        ProceduralParam { name: "low_material".into(), value: ProceduralParamValue::Material(p.low_material), range: None },
        ProceduralParam { name: "low_to_mid".into(), value: ProceduralParamValue::Float(p.low_to_mid), range: Some((-2.0, 2.0)) },
        ProceduralParam { name: "mid_material".into(), value: ProceduralParamValue::Material(p.mid_material), range: None },
        ProceduralParam { name: "mid_to_high".into(), value: ProceduralParamValue::Float(p.mid_to_high), range: Some((-2.0, 2.0)) },
        ProceduralParam { name: "high_material".into(), value: ProceduralParamValue::Material(p.high_material), range: None },
        ProceduralParam { name: "transition_width".into(), value: ProceduralParamValue::Float(p.transition_width), range: Some((0.0, 2.0)) },
        ProceduralParam { name: "frequency".into(), value: ProceduralParamValue::Float(p.frequency), range: Some((0.05, 32.0)) },
        ProceduralParam { name: "octaves".into(), value: ProceduralParamValue::Float(p.octaves as f32), range: Some((1.0, 8.0)) },
        ProceduralParam { name: "seed".into(), value: ProceduralParamValue::Float(p.seed as f32), range: Some((0.0, 1024.0)) },
    ]
}

fn color_by_noise_params(
    p: &rkp_procedural::node_kind::ColorByNoiseParams,
) -> Vec<ProceduralParam> {
    vec![
        ProceduralParam { name: "low_color".into(), value: ProceduralParamValue::Color(rgba_from_color(p.low_color)), range: None },
        ProceduralParam { name: "low_to_mid".into(), value: ProceduralParamValue::Float(p.low_to_mid), range: Some((-2.0, 2.0)) },
        ProceduralParam { name: "mid_color".into(), value: ProceduralParamValue::Color(rgba_from_color(p.mid_color)), range: None },
        ProceduralParam { name: "mid_to_high".into(), value: ProceduralParamValue::Float(p.mid_to_high), range: Some((-2.0, 2.0)) },
        ProceduralParam { name: "high_color".into(), value: ProceduralParamValue::Color(rgba_from_color(p.high_color)), range: None },
        ProceduralParam { name: "transition_width".into(), value: ProceduralParamValue::Float(p.transition_width), range: Some((0.0, 2.0)) },
        ProceduralParam { name: "frequency".into(), value: ProceduralParamValue::Float(p.frequency), range: Some((0.05, 32.0)) },
        ProceduralParam { name: "octaves".into(), value: ProceduralParamValue::Float(p.octaves as f32), range: Some((1.0, 8.0)) },
        ProceduralParam { name: "seed".into(), value: ProceduralParamValue::Float(p.seed as f32), range: Some((0.0, 1024.0)) },
    ]
}

fn mirror_params(
    p: &rkp_procedural::node_kind::MirrorParams,
) -> Vec<ProceduralParam> {
    use rkp_procedural::node_kind::MirrorAxis;
    let axis_value = match p.axis {
        MirrorAxis::X => "X",
        MirrorAxis::Y => "Y",
        MirrorAxis::Z => "Z",
    };
    // The mirror plane passes through the Mirror node's local origin
    // — move / rotate the node's transform to position it in world
    // space. No per-params position field, consistent with leaves
    // (their centers come from the transform too).
    vec![ProceduralParam {
        name: "axis".into(),
        value: ProceduralParamValue::Select {
            value: axis_value.into(),
            options: vec![
                ("X".into(), "X".into()),
                ("Y".into(), "Y".into()),
                ("Z".into(), "Z".into()),
            ],
        },
        range: None,
    }]
}

fn array_params(
    p: &rkp_procedural::node_kind::ArrayParams,
) -> Vec<ProceduralParam> {
    // Exposed as six scalars: three counts (rendered as integers via
    // the Float widget's whole-number range + round-at-apply) and
    // three per-axis spacings. A future int-param type could group
    // these into a proper `IntVec3` / `Vec3` pair, but scalar-per-
    // axis is what the existing prop_controls can render without
    // new widget work.
    let count = |label: &str, v: u32| ProceduralParam {
        name: label.into(),
        value: ProceduralParamValue::Float(v as f32),
        range: Some((1.0, 64.0)),
    };
    let spacing = |label: &str, v: f32| ProceduralParam {
        name: label.into(),
        value: ProceduralParamValue::Float(v),
        range: Some((0.01, 100.0)),
    };
    vec![
        count("count_x", p.counts[0]),
        count("count_y", p.counts[1]),
        count("count_z", p.counts[2]),
        spacing("spacing_x", p.spacings[0]),
        spacing("spacing_y", p.spacings[1]),
        spacing("spacing_z", p.spacings[2]),
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
