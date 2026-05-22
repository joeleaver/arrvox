//! Simulation component entries: RigidBody, Skeleton, AnimationPlayer.

use crate::inspector::{FieldType, FieldValue};

use super::{ComponentEntry, FieldMeta};

// ── RigidBody ───────────────────────────────────────────────────────

static RIGID_BODY_FIELDS: [FieldMeta; 6] = [
    FieldMeta { name: "body_type", field_type: FieldType::String, range: None, transient: false, struct_fields: None, asset_filter: None,
        enum_options: Some(&[("Dynamic", "Dynamic"), ("Static", "Static"), ("KinematicPosition", "Kinematic Pos"), ("KinematicVelocity", "Kinematic Vel")]), scrub: false },
    FieldMeta { name: "collider_shape", field_type: FieldType::String, range: None, transient: false, struct_fields: None, asset_filter: None,
        enum_options: Some(&[("Auto", "Auto (Voxel)"), ("Box", "Box"), ("Sphere", "Sphere"), ("Capsule", "Capsule")]), scrub: false },
    FieldMeta { name: "mass", field_type: FieldType::Float, range: Some((0.01, 1000.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
    FieldMeta { name: "friction", field_type: FieldType::Float, range: Some((0.0, 2.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
    FieldMeta { name: "restitution", field_type: FieldType::Float, range: Some((0.0, 1.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
    FieldMeta { name: "collider_cell_size", field_type: FieldType::Float, range: Some((0.05, 1.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: true },
];

pub(super) fn rigid_body_entry() -> ComponentEntry {
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
                            "Static" => arvx_physics::rigid_body::BodyType::Static,
                            "KinematicPosition" => arvx_physics::rigid_body::BodyType::KinematicPosition,
                            "KinematicVelocity" => arvx_physics::rigid_body::BodyType::KinematicVelocity,
                            _ => arvx_physics::rigid_body::BodyType::Dynamic,
                        };
                        Ok(())
                    } else { Err("type mismatch".into()) }
                }
                "collider_shape" => {
                    if let FieldValue::String(v) = value {
                        c.collider_shape = match v.as_str() {
                            "Box" => arvx_physics::rigid_body::ColliderShape::Box,
                            "Sphere" => arvx_physics::rigid_body::ColliderShape::Sphere,
                            "Capsule" => arvx_physics::rigid_body::ColliderShape::Capsule,
                            _ => arvx_physics::rigid_body::ColliderShape::Auto,
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
                    if let FieldValue::Float(v) = value {
                        // Clamp to slider range. The lower bound matters: voxel
                        // counts grow as 1/cs³, so unbounded values quickly
                        // exceed the GPU's max_buffer_size for the wireframe.
                        c.collider_cell_size = (v as f32).clamp(0.05, 1.0);
                        Ok(())
                    } else { Err("type mismatch".into()) }
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
        field_visible: None,
    }
}

// ── Skeleton ────────────────────────────────────────────────────────
//
// Transient — rebuilt from `.arvxskel` on entity load, never serialized to
// the scene file. Reflection exposes the loaded asset's shape (bone
// count, clip count, path) read-only so the inspector can surface it.

static SKELETON_FIELDS: [FieldMeta; 3] = [
    FieldMeta { name: "path", field_type: FieldType::String, range: None, transient: true, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
    FieldMeta { name: "bone_count", field_type: FieldType::Int, range: None, transient: true, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
    FieldMeta { name: "clip_count", field_type: FieldType::Int, range: None, transient: true, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
];

pub(super) fn skeleton_entry() -> ComponentEntry {
    use crate::components::Skeleton;
    ComponentEntry {
        name: "Skeleton",
        meta: &SKELETON_FIELDS,
        mandatory: false,
        has: |world, entity| world.get::<&Skeleton>(entity).is_ok(),
        get_field: |world, entity, field| {
            let c = world.get::<&Skeleton>(entity).map_err(|_| "no Skeleton".to_string())?;
            match field {
                "path" => Ok(FieldValue::String(c.path.to_string_lossy().into_owned())),
                "bone_count" => Ok(FieldValue::Int(c.asset.skeleton.bones.len() as i64)),
                "clip_count" => Ok(FieldValue::Int(c.asset.clips.len() as i64)),
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        // Every field is transient; the inspector never writes to a Skeleton.
        set_field: |_, _, field, _| Err(format!("Skeleton field '{field}' is read-only")),
        // `add_default` is a pass-through no-op. `AddComponent` for
        // "Skeleton" is special-cased by the engine command handler
        // (see `EngineCommand::AddComponent`) so it can reach the
        // sibling `.arvxskel` and the animation asset cache — the
        // registry's plain (World, Entity) signature isn't rich enough
        // to do the real attach here.
        add_default: |_, _| Ok(()),
        remove: |world, entity| {
            world.remove_one::<Skeleton>(entity).map(|_| ()).map_err(|e| format!("{e}"))
        },
        // Serialize only a presence marker — the actual skeleton asset
        // is rediscovered on load by finding the `.arvxskel` sibling of
        // the Renderable's asset. Emitting the stored path lets humans
        // inspect the scene file but isn't authoritative.
        serialize: |world, entity| {
            let c = world.get::<&Skeleton>(entity).ok()?;
            Some(serde_json::to_string(&c.path.to_string_lossy().into_owned())
                .unwrap_or_else(|_| "\"\"".to_string()))
        },
        // The real attach needs the asset cache + Renderable lookup,
        // which registry's `(world, entity, &str)` signature can't
        // reach. Engine handles `"Skeleton"` specially during scene
        // load (see `load_scene`) and this function is never called.
        deserialize_insert: |_, _, _| Err(
            "Skeleton deserialization is special-cased by scene load; this path should be unreachable".into()
        ),
        on_add: None,
        on_remove: None,
        field_visible: None,
    }
}

// ── AnimationPlayer ────────────────────────────────────────────────

static ANIMATION_PLAYER_FIELDS: [FieldMeta; 5] = [
    // clip_name: free-form string here; the editor panel renders a
    // dropdown by reading the sibling Skeleton's clip list directly.
    FieldMeta { name: "clip_name", field_type: FieldType::String, range: None, transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
    FieldMeta { name: "time", field_type: FieldType::Float, range: Some((0.0, 60.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: true },
    FieldMeta { name: "speed", field_type: FieldType::Float, range: Some((-4.0, 4.0)), transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: true },
    FieldMeta { name: "playing", field_type: FieldType::Bool, range: None, transient: false, struct_fields: None, asset_filter: None, enum_options: None, scrub: false },
    FieldMeta { name: "loop_mode", field_type: FieldType::String, range: None, transient: false, struct_fields: None, asset_filter: None,
        enum_options: Some(&[("Once", "Once"), ("Loop", "Loop"), ("PingPong", "PingPong")]), scrub: false },
];

pub(super) fn animation_player_entry() -> ComponentEntry {
    use crate::components::AnimationPlayer;
    use arvx_animation::player::LoopMode;
    ComponentEntry {
        name: "AnimationPlayer",
        meta: &ANIMATION_PLAYER_FIELDS,
        mandatory: false,
        has: |world, entity| world.get::<&AnimationPlayer>(entity).is_ok(),
        get_field: |world, entity, field| {
            let c = world.get::<&AnimationPlayer>(entity).map_err(|_| "no AnimationPlayer".to_string())?;
            match field {
                "clip_name" => Ok(FieldValue::String(c.clip_name.clone())),
                "time" => Ok(FieldValue::Float(c.time as f64)),
                "speed" => Ok(FieldValue::Float(c.speed as f64)),
                "playing" => Ok(FieldValue::Bool(c.playing)),
                "loop_mode" => Ok(FieldValue::String(match c.loop_mode {
                    LoopMode::Once => "Once".into(),
                    LoopMode::Loop => "Loop".into(),
                    LoopMode::PingPong => "PingPong".into(),
                })),
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        set_field: |world, entity, field, value| {
            let mut c = world.get::<&mut AnimationPlayer>(entity).map_err(|_| "no AnimationPlayer".to_string())?;
            match field {
                "clip_name" => {
                    if let FieldValue::String(v) = value { c.clip_name = v; c.time = 0.0; c.forward = true; Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                "time" => {
                    if let FieldValue::Float(v) = value { c.time = v as f32; Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                "speed" => {
                    if let FieldValue::Float(v) = value { c.speed = v as f32; Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                "playing" => {
                    if let FieldValue::Bool(v) = value { c.playing = v; Ok(()) }
                    else { Err("type mismatch".into()) }
                }
                "loop_mode" => {
                    if let FieldValue::String(v) = value {
                        c.loop_mode = match v.as_str() {
                            "Once" => LoopMode::Once,
                            "PingPong" => LoopMode::PingPong,
                            _ => LoopMode::Loop,
                        };
                        Ok(())
                    } else { Err("type mismatch".into()) }
                }
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        add_default: |world, entity| {
            world.insert_one(entity, AnimationPlayer::default()).map_err(|e| format!("{e}"))
        },
        remove: |world, entity| {
            world.remove_one::<AnimationPlayer>(entity).map(|_| ()).map_err(|e| format!("{e}"))
        },
        serialize: |world, entity| {
            let c = world.get::<&AnimationPlayer>(entity).ok()?;
            serde_json::to_string(&*c).ok()
        },
        deserialize_insert: |world, entity, json| {
            let c: AnimationPlayer = serde_json::from_str(json).map_err(|e| format!("{e}"))?;
            world.insert_one(entity, c).map_err(|e| format!("{e}"))
        },
        on_add: None,
        on_remove: None,
        field_visible: None,
    }
}
