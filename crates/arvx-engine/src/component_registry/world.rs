//! World/spatial component entries: ProceduralGeometry, Transform, EditorMetadata.

use crate::inspector::{FieldType, FieldValue};

use super::{ComponentEntry, FieldMeta};

// ── ProceduralGeometry ───────────────────────────────────────────────

/// Small surface: the tree itself is edited via the build panel, not the
/// inspector. We expose only the two scalars that make sense to tweak
/// without going through the build panel (voxel size tier, collider
/// resolution). Registration exists primarily so `ProceduralGeometry`
/// participates in `.arvxproject` save / load — without it, procedurals
/// silently lose their tree on reopen.
/// Pick the closest-matching `PROCEDURAL_VOXEL_TIERS` entry for a
/// stored voxel_size. Used by `get_field` so the inspector dropdown
/// shows whichever tier is closest when a save file carries a stray
/// value (legacy / external edit).
fn tier_string(voxel_size: f32) -> String {
    use crate::components::PROCEDURAL_VOXEL_TIERS;
    PROCEDURAL_VOXEL_TIERS
        .iter()
        .min_by(|a, b| {
            let da = (a.0.parse::<f32>().unwrap_or(f32::NAN) - voxel_size).abs();
            let db = (b.0.parse::<f32>().unwrap_or(f32::NAN) - voxel_size).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(v, _)| (*v).to_string())
        .unwrap_or_else(|| "0.02".to_string())
}

/// Snap a parsed voxel_size to the nearest tier. Mirrors
/// `SetProceduralVoxelSize`'s logic so either code path produces the
/// same value on `proc_geo.voxel_size`.
fn snap_to_tier(voxel_size: f32) -> f32 {
    use crate::components::PROCEDURAL_VOXEL_TIERS;
    PROCEDURAL_VOXEL_TIERS
        .iter()
        .filter_map(|(v, _)| v.parse::<f32>().ok())
        .min_by(|a, b| {
            (a - voxel_size).abs()
                .partial_cmp(&(b - voxel_size).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0.02)
}

static PROCEDURAL_FIELDS: [FieldMeta; 1] = [
    // Enum picker backed by `PROCEDURAL_VOXEL_TIERS` — same tier
    // labels the build viewport's resolution dropdown shows. Picking
    // either surface writes to `proc_geo.voxel_size`; the snap in
    // `SetProceduralVoxelSize` + the parse here keep both paths in
    // lockstep.
    FieldMeta { name: "voxel_size", field_type: FieldType::String, range: None, transient: false, struct_fields: None, asset_filter: None,
        enum_options: Some(crate::components::PROCEDURAL_VOXEL_TIERS), scrub: false },
];

pub(super) fn procedural_geometry_entry() -> ComponentEntry {
    use crate::components::ProceduralGeometry;
    ComponentEntry {
        name: "ProceduralGeometry",
        meta: &PROCEDURAL_FIELDS,
        mandatory: false,
        has: |world, entity| world.get::<&ProceduralGeometry>(entity).is_ok(),
        get_field: |world, entity, field| {
            let c = world.get::<&ProceduralGeometry>(entity).map_err(|_| "no ProceduralGeometry".to_string())?;
            match field {
                // Return the exact tier string so the enum picker
                // lines up with the matching `(value, label)` entry.
                "voxel_size" => Ok(FieldValue::String(tier_string(c.voxel_size))),
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        set_field: |world, entity, field, value| {
            let mut c = world.get::<&mut ProceduralGeometry>(entity).map_err(|_| "no ProceduralGeometry".to_string())?;
            match field {
                "voxel_size" => {
                    // Parse + snap to nearest tier so typos / stale
                    // saved values round to something valid; mirrors
                    // the logic in `SetProceduralVoxelSize`.
                    if let FieldValue::String(v) = value {
                        let Ok(parsed) = v.parse::<f32>() else {
                            return Err(format!("invalid voxel_size '{v}'"));
                        };
                        let snapped = snap_to_tier(parsed);
                        if (snapped - c.voxel_size).abs() > 1e-6 {
                            c.voxel_size = snapped;
                            // Auto-bake on voxel-size change — same
                            // rationale as `SetProceduralVoxelSize`.
                            c.pending_bake = true;
                            c.bake_dirty_at = Some(std::time::Instant::now());
                        }
                        Ok(())
                    } else { Err("type mismatch".into()) }
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
            let mut c: ProceduralGeometry = serde_json::from_str(json).map_err(|e| format!("{e}"))?;
            // Legacy-scene migration: early prototypes stored a bare
            // leaf (usually Sphere) as the root. The new model makes
            // the root a `NodeKind::Root` container. Wrap legacy
            // leaf-roots in a Root so saved scenes keep working.
            // Union/Intersect/Subtract roots are already valid
            // containers and pass through untouched.
            let root_id = c.tree.root();
            let needs_wrap = c
                .tree
                .get(root_id)
                .map(|n| n.kind.is_leaf())
                .unwrap_or(false);
            if needs_wrap {
                c.tree.wrap_in_root();
            }
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
    // Upper bound matches the per-axis cap enforced in
    // `redirect_transform_scale_to_root` for procedural entities.
    // Non-procedurals (lights, cameras, imported meshes) aren't
    // clamped but this range still makes the slider usable — no
    // reason to default the UI range to something nobody hits.
    FieldMeta { name: "scale", field_type: FieldType::Vec3, range: Some((0.01, 20.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
];

pub(super) fn transform_entry() -> ComponentEntry {
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

pub(super) fn editor_metadata_entry() -> ComponentEntry {
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

