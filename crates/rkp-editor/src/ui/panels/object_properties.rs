//! Object properties panel — displays and edits selected object's properties.
//!
//! Shows: name, entity ID, transform (position/rotation/scale),
//! and dynamic ECS component fields from the inspector snapshot.

use rinch::prelude::*;

use crate::CommandSender;
use crate::ui::store::EditorStore;
use rkp_engine::inspector::*;

#[component]
pub fn ObjectProperties() -> NodeHandle {
    let store = use_context::<EditorStore>();

    rsx! {
        div {
            style: "display:flex;flex-direction:column;height:100%;overflow-y:auto;\
                    color:#ccc;font-size:12px;",
            if store.inspector.get().is_some() {
                InspectorContent {}
            }
            if store.inspector.get().is_none() {
                div {
                    style: "padding:16px;color:#666;font-style:italic;text-align:center;",
                    {"Select an object to view its properties"}
                }
            }
        }
    }
}

#[component]
fn InspectorContent() -> NodeHandle {
    let store = use_context::<EditorStore>();

    rsx! {
        div {
            style: "display:flex;flex-direction:column;",

            // Header: name + entity ID
            div {
                style: "padding:8px 12px;background:#2d2d2d;border-bottom:1px solid #3c3c3c;",
                div {
                    style: "font-weight:600;font-size:13px;",
                    {|| store.inspector.get().map(|i| i.entity_name.clone()).unwrap_or_default()}
                }
                div {
                    style: "font-size:10px;color:#666;margin-top:2px;font-family:monospace;",
                    {|| {
                        let id = store.inspector.get()
                            .map(|i| i.entity_id.clone())
                            .unwrap_or_default();
                        id.chars().take(8).collect::<String>()
                    }}
                }
            }

            // Transform section
            div {
                style: "padding:8px 12px;",
                SectionHeader { title: "Transform".to_string() }
                TransformRow { label: "Position".to_string(), field: "position".to_string() }
                TransformRow { label: "Rotation".to_string(), field: "rotation".to_string() }
                TransformRow { label: "Scale".to_string(), field: "scale".to_string() }
            }

            // Dynamic ECS components
            ComponentsList {}
        }
    }
}

// ── Reusable section header ──────────────────────────────────────────

#[component]
fn SectionHeader(title: String) -> NodeHandle {
    rsx! {
        div {
            style: "font-size:10px;font-weight:600;color:#888;text-transform:uppercase;\
                    letter-spacing:0.5px;margin-bottom:6px;padding-bottom:4px;\
                    border-bottom:1px solid #3c3c3c;",
            {title}
        }
    }
}

// ── Transform editing ────────────────────────────────────────────────

/// A row with label + 3 numeric inputs for a transform vec3 field.
/// `field` is one of "position", "rotation", "scale".
#[component]
fn TransformRow(label: String, field: String) -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd = use_context::<CommandSender>();

    // Read the current vec3 value from the inspector snapshot.
    let vec3 = Memo::new({
        let field = field.clone();
        move || -> [f32; 3] {
            let Some(snap) = store.inspector.get() else { return [0.0; 3] };
            match field.as_str() {
                "position" => snap.position,
                "rotation" => snap.rotation,
                "scale" => snap.scale,
                _ => [0.0; 3],
            }
        }
    });

    // Helper to send a transform update for one changed axis.
    let send_axis = {
        let field = field.clone();
        let cmd = cmd.clone();
        move |axis_idx: usize, new_val: f32| {
            let mut v = vec3.get();
            v[axis_idx] = new_val;
            let Some(snap) = store.inspector.get() else { return };
            let Ok(entity_id) = uuid::Uuid::parse_str(&snap.entity_id) else { return };
            let vec = glam::Vec3::from_array(v);
            let command = match field.as_str() {
                "position" => rkp_engine::EngineCommand::SetObjectPosition { entity_id, position: vec },
                "rotation" => rkp_engine::EngineCommand::SetObjectRotation { entity_id, rotation: vec },
                "scale" => rkp_engine::EngineCommand::SetObjectScale { entity_id, scale: vec },
                _ => return,
            };
            let _ = cmd.0.send(command);
        }
    };

    rsx! {
        div {
            style: "display:flex;align-items:center;margin-bottom:4px;gap:4px;",
            div { style: "width:60px;flex-shrink:0;color:#888;font-size:11px;", {label} }
            div {
                style: "display:flex;flex:1;gap:2px;",
                AxisInput { color: "#e06060".to_string(), name: "X".to_string(), component_idx: "0".to_string(), transform_field: field.clone() }
                AxisInput { color: "#60e060".to_string(), name: "Y".to_string(), component_idx: "1".to_string(), transform_field: field.clone() }
                AxisInput { color: "#6060e0".to_string(), name: "Z".to_string(), component_idx: "2".to_string(), transform_field: field.clone() }
            }
        }
    }
}

/// Single axis input for a transform vec3 field. Uses string props to
/// work with rinch's component system (which wraps numeric props in Option).
#[component]
fn AxisInput(color: String, name: String, component_idx: String, transform_field: String) -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd = use_context::<CommandSender>();
    let idx: usize = component_idx.parse().unwrap_or(0);

    let get_vec3 = Memo::new({
        let field = transform_field.clone();
        move || -> [f32; 3] {
            let Some(snap) = store.inspector.get() else { return [0.0; 3] };
            match field.as_str() {
                "position" => snap.position,
                "rotation" => snap.rotation,
                "scale" => snap.scale,
                _ => [0.0; 3],
            }
        }
    });

    rsx! {
        div {
            style: "flex:1;display:flex;align-items:center;",
            span {
                style: {|| format!("color:{};font-size:10px;font-weight:600;margin-right:2px;", color)},
                {name}
            }
            input {
                r#type: "number",
                style: "width:100%;background:#1e1e1e;border:1px solid #3c3c3c;\
                        color:#ccc;padding:2px 4px;font-size:11px;border-radius:2px;outline:none;",
                value: move || format!("{:.2}", get_vec3.get()[idx]),
                oninput: {
                    let field = transform_field.clone();
                    let cmd = cmd.clone();
                    move |val: String| {
                        if let Ok(v) = val.parse::<f32>() {
                            let mut vec = get_vec3.get();
                            vec[idx] = v;
                            let Some(snap) = store.inspector.get() else { return };
                            let Ok(entity_id) = uuid::Uuid::parse_str(&snap.entity_id) else { return };
                            let glam_vec = glam::Vec3::from_array(vec);
                            let command = match field.as_str() {
                                "position" => rkp_engine::EngineCommand::SetObjectPosition { entity_id, position: glam_vec },
                                "rotation" => rkp_engine::EngineCommand::SetObjectRotation { entity_id, rotation: glam_vec },
                                "scale" => rkp_engine::EngineCommand::SetObjectScale { entity_id, scale: glam_vec },
                                _ => return,
                            };
                            let _ = cmd.0.send(command);
                        }
                    }
                },
            }
        }
    }
}

// ── Dynamic component list ───────────────────────────────────────────

/// Renders all ECS components (except Transform which has its own section).
#[component]
fn ComponentsList() -> NodeHandle {
    let store = use_context::<EditorStore>();

    let components = Memo::new(move || {
        store.inspector.get()
            .map(|snap| snap.components.into_iter()
                .filter(|c| c.name != "Transform")
                .collect::<Vec<_>>())
            .unwrap_or_default()
    });

    rsx! {
        div {
            style: "padding:0 12px 12px;",
            for comp in components.get() {
                div {
                    key: comp.name.clone(),
                    style: "margin-top:8px;",
                    SectionHeader { title: comp.name.clone() }
                    for field in comp.fields.clone() {
                        FieldRow {
                            key: field.name.clone(),
                            snapshot: field.clone(),
                            component: comp.name.clone(),
                        }
                    }
                }
            }
        }
    }
}

/// Renders a single component field with a type-appropriate editor.
#[component]
fn FieldRow(snapshot: FieldSnapshot, component: String) -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd = use_context::<CommandSender>();
    let field_name = snapshot.name.clone();

    rsx! {
        div {
            style: "display:flex;align-items:center;margin-bottom:3px;gap:4px;",
            div {
                style: "width:80px;flex-shrink:0;color:#888;font-size:11px;",
                {field_name.clone()}
            }
            div {
                style: "flex:1;",
                {field_editor(
                    __scope,
                    &snapshot,
                    &component,
                    &field_name,
                    store,
                    cmd.clone(),
                )}
            }
        }
    }
}

/// Build the appropriate editor widget for a field based on its type.
fn field_editor(
    __scope: &mut rinch::core::dom::RenderScope,
    snapshot: &FieldSnapshot,
    component: &str,
    field_name: &str,
    store: EditorStore,
    cmd: CommandSender,
) -> rinch::core::dom::NodeHandle {
    match snapshot.field_type {
        FieldType::Float => {
            let val = match &snapshot.value { FieldValue::Float(v) => *v, _ => 0.0 };
            let comp = component.to_string();
            let fname = field_name.to_string();
            rsx! {
                input {
                    r#type: "number",
                    style: "width:100%;background:#1e1e1e;border:1px solid #3c3c3c;\
                            color:#ccc;padding:2px 4px;font-size:11px;border-radius:2px;",
                    value: {|| format!("{val:.3}")},
                    oninput: move |v: String| {
                        if let Ok(fv) = v.parse::<f64>() {
                            send_field(&cmd, &store, &comp, &fname, FieldValue::Float(fv));
                        }
                    },
                }
            }
        }
        FieldType::Int => {
            let val = match &snapshot.value { FieldValue::Int(v) => *v, _ => 0 };
            let comp = component.to_string();
            let fname = field_name.to_string();
            rsx! {
                input {
                    r#type: "number",
                    style: "width:100%;background:#1e1e1e;border:1px solid #3c3c3c;\
                            color:#ccc;padding:2px 4px;font-size:11px;border-radius:2px;",
                    value: {|| format!("{val}")},
                    oninput: move |v: String| {
                        if let Ok(iv) = v.parse::<i64>() {
                            send_field(&cmd, &store, &comp, &fname, FieldValue::Int(iv));
                        }
                    },
                }
            }
        }
        FieldType::Bool => {
            let val = match &snapshot.value { FieldValue::Bool(v) => *v, _ => false };
            let comp = component.to_string();
            let fname = field_name.to_string();
            let checked = Signal::new(val);
            rsx! {
                div {
                    style: {move || {
                        if checked.get() {
                            "width:16px;height:16px;border-radius:3px;cursor:pointer;\
                             background:#007acc;border:1px solid #007acc;"
                        } else {
                            "width:16px;height:16px;border-radius:3px;cursor:pointer;\
                             background:#1e1e1e;border:1px solid #3c3c3c;"
                        }
                    }},
                    onclick: move || {
                        let new_val = !checked.get();
                        checked.set(new_val);
                        send_field(&cmd, &store, &comp, &fname, FieldValue::Bool(new_val));
                    },
                }
            }
        }
        FieldType::String => {
            let val = match &snapshot.value { FieldValue::String(v) => v.clone(), _ => String::new() };
            let comp = component.to_string();
            let fname = field_name.to_string();
            rsx! {
                input {
                    r#type: "text",
                    style: "width:100%;background:#1e1e1e;border:1px solid #3c3c3c;\
                            color:#ccc;padding:2px 4px;font-size:11px;border-radius:2px;",
                    value: {|| val.clone()},
                    oninput: move |v: String| {
                        send_field(&cmd, &store, &comp, &fname, FieldValue::String(v));
                    },
                }
            }
        }
        _ => {
            // Fallback: display value as text.
            let display = snapshot.value.to_string();
            rsx! {
                div {
                    style: "color:#ccc;font-size:11px;font-family:monospace;\
                            overflow:hidden;text-overflow:ellipsis;white-space:nowrap;",
                    {display}
                }
            }
        }
    }
}

/// Send a SetComponentField command to the engine.
fn send_field(
    cmd: &CommandSender,
    store: &EditorStore,
    component: &str,
    field: &str,
    value: FieldValue,
) {
    let Some(snap) = store.inspector.get() else { return };
    let Ok(entity_id) = uuid::Uuid::parse_str(&snap.entity_id) else { return };
    let serialized = serde_json::to_string(&value).unwrap_or_default();
    let _ = cmd.0.send(rkp_engine::EngineCommand::SetComponentField {
        entity_id,
        component_name: component.to_string(),
        field_name: field.to_string(),
        value: serialized,
    });
}
