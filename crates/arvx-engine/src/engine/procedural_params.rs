//! Procedural node parameter coercion and leaf-kind parsing.
//!
//! `apply_procedural_param` takes a JSON value from the editor and
//! writes it into the appropriate `NodeKind` variant field on the
//! target node, doing type coercion + validation along the way. The
//! smaller helpers (`parse_node_kind`, `parse_vec3`,
//! `json_to_field_value`) share that path and the generator-run preset
//! path in `generator_ops`.


/// Collect all leaf voxel-pool slots from an octree in the packed node buffer.
///
/// Branch offsets in the packed buffer are ABSOLUTE indices. This function
/// traverses from `node_idx` directly in `all_nodes` without sub-slicing,
/// avoiding the offset-rebasing problem that `SparseOctree::from_raw` has
/// when given a sub-slice.
/// Coerce a JSON preset value into a `FieldValue` whose variant matches
/// the field's declared `FieldType`. Returns a descriptive error if the
/// types don't line up — the caller logs and continues.
pub(crate) fn json_to_field_value(
    value: &serde_json::Value,
    field_name: &str,
    comp: &crate::component_registry::ComponentEntry,
) -> Result<crate::inspector::FieldValue, String> {
    use crate::inspector::{FieldType, FieldValue};
    let meta = comp
        .meta
        .iter()
        .find(|m| m.name == field_name)
        .ok_or_else(|| format!("unknown field '{field_name}'"))?;
    match meta.field_type {
        FieldType::Float => value
            .as_f64()
            .map(FieldValue::Float)
            .ok_or_else(|| format!("expected number for {field_name}")),
        FieldType::Int => value
            .as_i64()
            .map(FieldValue::Int)
            .ok_or_else(|| format!("expected integer for {field_name}")),
        FieldType::Bool => value
            .as_bool()
            .map(FieldValue::Bool)
            .ok_or_else(|| format!("expected boolean for {field_name}")),
        FieldType::String => value
            .as_str()
            .map(|s| FieldValue::String(s.to_string()))
            .ok_or_else(|| format!("expected string for {field_name}")),
        FieldType::Vec3 => {
            let arr = value
                .as_array()
                .filter(|a| a.len() == 3)
                .ok_or_else(|| format!("expected [x,y,z] for {field_name}"))?;
            let mut out = [0.0f32; 3];
            for (i, v) in arr.iter().enumerate() {
                out[i] = v
                    .as_f64()
                    .ok_or_else(|| format!("non-number in {field_name}[{i}]"))?
                    as f32;
            }
            Ok(FieldValue::Vec3(out))
        }
        FieldType::Color => {
            let arr = value
                .as_array()
                .filter(|a| a.len() == 4)
                .ok_or_else(|| format!("expected [r,g,b,a] for {field_name}"))?;
            let mut out = [0.0f32; 4];
            for (i, v) in arr.iter().enumerate() {
                out[i] = v
                    .as_f64()
                    .ok_or_else(|| format!("non-number in {field_name}[{i}]"))?
                    as f32;
            }
            Ok(FieldValue::Color(out))
        }
    }
}

/// Parse a node kind name into a `NodeKind`.
pub(crate) fn parse_node_kind(kind: &str) -> arvx_procedural::NodeKind {
    use arvx_procedural::node_kind::*;
    match kind {
        "Sphere" => arvx_procedural::NodeKind::Sphere(SphereParams::default()),
        "Box" => arvx_procedural::NodeKind::Box(BoxParams::default()),
        "Capsule" => arvx_procedural::NodeKind::Capsule(CapsuleParams::default()),
        "Cylinder" => arvx_procedural::NodeKind::Cylinder(CylinderParams::default()),
        "Torus" => arvx_procedural::NodeKind::Torus(TorusParams::default()),
        "Plane" => arvx_procedural::NodeKind::Plane(PlaneParams::default()),
        "Ramp" => arvx_procedural::NodeKind::Ramp(RampParams::default()),
        "Union" => arvx_procedural::NodeKind::Union {
            material_combine: arvx_procedural::MaterialCombine::Winner,
        },
        "Intersect" => arvx_procedural::NodeKind::Intersect {
            material_combine: arvx_procedural::MaterialCombine::Winner,
        },
        "Subtract" => arvx_procedural::NodeKind::Subtract,
        "NoiseDisplace" => {
            arvx_procedural::NodeKind::NoiseDisplace(NoiseDisplaceParams::default())
        }
        "Mirror" => arvx_procedural::NodeKind::Mirror(MirrorParams::default()),
        "MaterialByHeight" => {
            arvx_procedural::NodeKind::MaterialByHeight(MaterialByHeightParams::default())
        }
        "ColorByHeight" => {
            arvx_procedural::NodeKind::ColorByHeight(ColorByHeightParams::default())
        }
        "MaterialByNoise" => {
            arvx_procedural::NodeKind::MaterialByNoise(MaterialByNoiseParams::default())
        }
        "ColorByNoise" => {
            arvx_procedural::NodeKind::ColorByNoise(ColorByNoiseParams::default())
        }
        "Array" => arvx_procedural::NodeKind::Array(ArrayParams::default()),
        _ => arvx_procedural::NodeKind::Sphere(SphereParams::default()),
    }
}

/// Apply a parameter value to a procedural node. Returns true if the param was found.
pub(crate) fn apply_procedural_param(
    tree: &mut arvx_procedural::ProceduralObject,
    id: arvx_procedural::NodeId,
    param_name: &str,
    value: &str,
) -> bool {
    use arvx_procedural::NodeKind;

    let node = match tree.get_mut(id) {
        Some(n) => n,
        None => return false,
    };

    match &mut node.kind {
        // Root has no editable params. Present a row with no fields
        // in the inspector; silently no-op any set attempts.
        NodeKind::Root => false,
        NodeKind::Sphere(p) => match param_name {
            "radius" => { p.radius = value.parse().unwrap_or(p.radius); true }
            "material_id" | "material" => { p.material_id = value.parse().unwrap_or(p.material_id); true }
            "color" => { if let Some(v) = parse_vec3(value) { p.color = v; } true }
            _ => false,
        },
        NodeKind::Box(p) => match param_name {
            "half_extents" => { if let Some(v) = parse_vec3(value) { p.half_extents = v; } true }
            "rounding" => { p.rounding = value.parse().unwrap_or(p.rounding); true }
            "material_id" | "material" => { p.material_id = value.parse().unwrap_or(p.material_id); true }
            "color" => { if let Some(v) = parse_vec3(value) { p.color = v; } true }
            _ => false,
        },
        NodeKind::Capsule(p) => match param_name {
            "half_height" => { p.half_height = value.parse().unwrap_or(p.half_height); true }
            "radius" => { p.radius = value.parse().unwrap_or(p.radius); true }
            "material_id" | "material" => { p.material_id = value.parse().unwrap_or(p.material_id); true }
            _ => false,
        },
        NodeKind::Cylinder(p) => match param_name {
            "half_height" => { p.half_height = value.parse().unwrap_or(p.half_height); true }
            "radius" => { p.radius = value.parse().unwrap_or(p.radius); true }
            "material_id" | "material" => { p.material_id = value.parse().unwrap_or(p.material_id); true }
            _ => false,
        },
        NodeKind::Torus(p) => match param_name {
            "major_radius" => { p.major_radius = value.parse().unwrap_or(p.major_radius); true }
            // `tube_radius` is the UI-visible name; `minor_radius` is kept
            // as an alias so the raw field name still works from MCP/scripts.
            "minor_radius" | "tube_radius" => { p.minor_radius = value.parse().unwrap_or(p.minor_radius); true }
            "material_id" | "material" => { p.material_id = value.parse().unwrap_or(p.material_id); true }
            _ => false,
        },
        NodeKind::Plane(p) => match param_name {
            "material_id" | "material" => { p.material_id = value.parse().unwrap_or(p.material_id); true }
            _ => false,
        },
        NodeKind::Ramp(p) => match param_name {
            "half_length" => { p.half_length = value.parse().unwrap_or(p.half_length); true }
            "half_height" => { p.half_height = value.parse().unwrap_or(p.half_height); true }
            "half_width" => { p.half_width = value.parse().unwrap_or(p.half_width); true }
            "material_id" | "material" => { p.material_id = value.parse().unwrap_or(p.material_id); true }
            "color" => { if let Some(v) = parse_vec3(value) { p.color = v; } true }
            _ => false,
        },
        NodeKind::Union { material_combine } | NodeKind::Intersect { material_combine } => {
            if param_name == "material_combine" {
                *material_combine = match value {
                    "Layered" => arvx_procedural::MaterialCombine::Layered,
                    "Blend" => arvx_procedural::MaterialCombine::Blend { radius: 0.1 },
                    _ => arvx_procedural::MaterialCombine::Winner,
                };
                true
            } else {
                false
            }
        }
        NodeKind::Subtract => false,
        NodeKind::NoiseDisplace(p) => match param_name {
            "amplitude" => { p.amplitude = value.parse().unwrap_or(p.amplitude); true }
            "frequency" => { p.frequency = value.parse().unwrap_or(p.frequency); true }
            // Octaves + seed come in as floats via the UI's Float
            // scrub control — round to u32 and clamp octaves to the
            // same bound `fbm_3d_vec` enforces so the stored value
            // matches what the evaluator actually uses.
            "octaves" => {
                let f: f32 = value.parse().unwrap_or(p.octaves as f32);
                p.octaves = (f.max(0.0) as u32).clamp(1, 8);
                true
            }
            "seed" => {
                let f: f32 = value.parse().unwrap_or(p.seed as f32);
                p.seed = f.max(0.0) as u32;
                true
            }
            _ => false,
        },
        NodeKind::Mirror(p) => match param_name {
            "axis" => {
                use arvx_procedural::node_kind::MirrorAxis;
                p.axis = match value {
                    "Y" => MirrorAxis::Y,
                    "Z" => MirrorAxis::Z,
                    _ => MirrorAxis::X,
                };
                true
            }
            _ => false,
        },
        NodeKind::MaterialByHeight(p) => match param_name {
            "low_material" => { p.low_material = value.parse().unwrap_or(p.low_material); true }
            "low_to_mid" => { p.low_to_mid = value.parse().unwrap_or(p.low_to_mid); true }
            "mid_material" => { p.mid_material = value.parse().unwrap_or(p.mid_material); true }
            "mid_to_high" => { p.mid_to_high = value.parse().unwrap_or(p.mid_to_high); true }
            "high_material" => { p.high_material = value.parse().unwrap_or(p.high_material); true }
            "transition_width" => {
                p.transition_width = value.parse::<f32>().unwrap_or(p.transition_width).max(0.0);
                true
            }
            _ => false,
        },
        NodeKind::ColorByHeight(p) => match param_name {
            "low_color" => { if let Some(v) = parse_vec3(value) { p.low_color = v; } true }
            "low_to_mid" => { p.low_to_mid = value.parse().unwrap_or(p.low_to_mid); true }
            "mid_color" => { if let Some(v) = parse_vec3(value) { p.mid_color = v; } true }
            "mid_to_high" => { p.mid_to_high = value.parse().unwrap_or(p.mid_to_high); true }
            "high_color" => { if let Some(v) = parse_vec3(value) { p.high_color = v; } true }
            "transition_width" => {
                p.transition_width = value.parse::<f32>().unwrap_or(p.transition_width).max(0.0);
                true
            }
            _ => false,
        },
        NodeKind::MaterialByNoise(p) => match param_name {
            "low_material" => { p.low_material = value.parse().unwrap_or(p.low_material); true }
            "low_to_mid" => { p.low_to_mid = value.parse().unwrap_or(p.low_to_mid); true }
            "mid_material" => { p.mid_material = value.parse().unwrap_or(p.mid_material); true }
            "mid_to_high" => { p.mid_to_high = value.parse().unwrap_or(p.mid_to_high); true }
            "high_material" => { p.high_material = value.parse().unwrap_or(p.high_material); true }
            "transition_width" => {
                p.transition_width = value.parse::<f32>().unwrap_or(p.transition_width).max(0.0);
                true
            }
            "frequency" => { p.frequency = value.parse().unwrap_or(p.frequency); true }
            "octaves" => {
                let f: f32 = value.parse().unwrap_or(p.octaves as f32);
                p.octaves = (f.max(0.0) as u32).clamp(1, 8);
                true
            }
            "seed" => {
                let f: f32 = value.parse().unwrap_or(p.seed as f32);
                p.seed = f.max(0.0) as u32;
                true
            }
            _ => false,
        },
        NodeKind::ColorByNoise(p) => match param_name {
            "low_color" => { if let Some(v) = parse_vec3(value) { p.low_color = v; } true }
            "low_to_mid" => { p.low_to_mid = value.parse().unwrap_or(p.low_to_mid); true }
            "mid_color" => { if let Some(v) = parse_vec3(value) { p.mid_color = v; } true }
            "mid_to_high" => { p.mid_to_high = value.parse().unwrap_or(p.mid_to_high); true }
            "high_color" => { if let Some(v) = parse_vec3(value) { p.high_color = v; } true }
            "transition_width" => {
                p.transition_width = value.parse::<f32>().unwrap_or(p.transition_width).max(0.0);
                true
            }
            "frequency" => { p.frequency = value.parse().unwrap_or(p.frequency); true }
            "octaves" => {
                let f: f32 = value.parse().unwrap_or(p.octaves as f32);
                p.octaves = (f.max(0.0) as u32).clamp(1, 8);
                true
            }
            "seed" => {
                let f: f32 = value.parse().unwrap_or(p.seed as f32);
                p.seed = f.max(0.0) as u32;
                true
            }
            _ => false,
        },
        NodeKind::Array(p) => {
            // Counts are per-axis u32s but the UI Float widget hands
            // us strings — round and clamp to ≥ 1 (0 would divide-by-
            // zero in the flatten emit).
            let set_count = |p_slot: &mut u32, v: &str| {
                let f: f32 = v.parse().unwrap_or(*p_slot as f32);
                *p_slot = (f.round().max(1.0) as u32).max(1);
            };
            match param_name {
                "count_x" => { set_count(&mut p.counts[0], value); true }
                "count_y" => { set_count(&mut p.counts[1], value); true }
                "count_z" => { set_count(&mut p.counts[2], value); true }
                "spacing_x" => {
                    p.spacings[0] = value.parse::<f32>().unwrap_or(p.spacings[0]).max(1e-4);
                    true
                }
                "spacing_y" => {
                    p.spacings[1] = value.parse::<f32>().unwrap_or(p.spacings[1]).max(1e-4);
                    true
                }
                "spacing_z" => {
                    p.spacings[2] = value.parse::<f32>().unwrap_or(p.spacings[2]).max(1e-4);
                    true
                }
                _ => false,
            }
        }
    }
}

pub(crate) fn parse_vec3(value: &str) -> Option<glam::Vec3> {
    // Accept "x,y,z" or "[x,y,z]"
    let cleaned = value.trim_matches(|c| c == '[' || c == ']' || c == ' ');
    let parts: Vec<f32> = cleaned.split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if parts.len() == 3 {
        Some(glam::Vec3::new(parts[0], parts[1], parts[2]))
    } else {
        None
    }
}
