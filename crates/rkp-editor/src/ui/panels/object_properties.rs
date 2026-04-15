//! Object properties panel — the heart of the editor.
//!
//! Displays and edits all components on the selected entity using the
//! component registry's reflection system. Components are shown as
//! collapsible sections with per-type field editors.

use std::rc::Rc;

use rinch::prelude::*;

use crate::CommandSender;
use crate::ui::store::EditorStore;
use rkp_engine::inspector::*;

use super::field_editors;
use super::prop_controls::*;

type CmdSignal = Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>;

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
                    style: "padding:24px 16px;color:#666;font-style:italic;text-align:center;",
                    {"Select an object to inspect its components"}
                }
            }
        }
    }
}

#[derive(Clone, PartialEq)]
struct KeyedComponent {
    entity_id: String,
    component: ComponentSnapshot,
}

fn flatten_keyed(keyed: (String, Vec<ComponentSnapshot>)) -> Vec<KeyedComponent> {
    let (entity_id, comps) = keyed;
    comps
        .into_iter()
        .map(|component| KeyedComponent {
            entity_id: entity_id.clone(),
            component,
        })
        .collect()
}

#[component]
fn InspectorContent() -> NodeHandle {
    let store = use_context::<EditorStore>();

    // Combine entity_id + components into a single memo so the for-loop's
    // effect subscribes to only one source. Subscribing to both memos caused
    // the for-loop effect to be re-queued while still running: whichever
    // memo's marker hadn't fired yet would queue the loop effect again,
    // leading to a RefCell re-entrancy panic on the trailing flush.
    let keyed_components = Memo::new(move || {
        store.inspector.get()
            .map(|snap| (snap.entity_id.clone(), snap.components.clone()))
            .unwrap_or_default()
    });

    let entity_name = Memo::new(move || {
        store.inspector.get().map(|s| s.entity_name.clone()).unwrap_or_default()
    });

    let entity_id = Memo::new(move || {
        store.inspector.get().map(|s| s.entity_id.clone()).unwrap_or_default()
    });

    rsx! {
        div {
            style: "display:flex;flex-direction:column;",

            // Entity header
            div {
                style: "padding:8px 12px;background:#2d2d2d;border-bottom:1px solid #3c3c3c;",
                div {
                    style: "font-weight:600;font-size:13px;",
                    {move || entity_name.get()}
                }
                div {
                    style: "font-size:10px;color:#666;margin-top:2px;font-family:monospace;",
                    {move || entity_id.get().chars().take(8).collect::<String>()}
                }
            }

            // Component sections.
            // Key includes entity_id so components remount when selection
            // changes. Reading entity_id + components from a single memo
            // (above) means this for-loop's effect only subscribes to that
            // one source — avoiding a re-entrancy crash when multiple memo
            // markers would each re-queue this effect.
            for keyed in flatten_keyed(keyed_components.get()) {
                ComponentSection {
                    key: format!("{}-{}", keyed.entity_id, keyed.component.name),
                    snapshot: keyed.component,
                }
            }

            // Material usage section (for entities with voxel data)
            MaterialUsageSection {}

            // Add Component button
            AddComponentButton {}
        }
    }
}

/// A single component section — collapsible header + field editors.
#[component]
fn ComponentSection(snapshot: ComponentSnapshot) -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd_tx: CmdSignal = Signal::new(use_context::<CommandSender>().0);
    let collapsed = Signal::new(false);
    let comp_name = Signal::new(snapshot.name.clone());
    let fields = Signal::new(snapshot.fields.clone());

    // Hide EditorMetadata — it's shown in the header.
    if snapshot.name == "EditorMetadata" {
        return rsx! { span {} };
    }

    // Build the remove callback for non-mandatory components.
    let on_remove = if snapshot.removable {
        Some(Rc::new(move || {
            let cn = comp_name.get();
            if let Some(snap) = store.inspector.get() {
                if let Ok(eid) = uuid::Uuid::parse_str(&snap.entity_id) {
                    let _ = cmd_tx.get().send(rkp_engine::EngineCommand::RemoveComponent {
                        entity_id: eid,
                        component_name: cn,
                    });
                }
            }
        }) as Rc<dyn Fn()>)
    } else {
        None
    };

    rsx! {
        div {
            {prop_section_header(__scope, &snapshot.name, collapsed, on_remove)}

            // Fields (hidden when collapsed)
            if !collapsed.get() {
                div {
                    style: "padding:6px 12px;display:flex;flex-direction:column;gap:2px;",
                    for field in fields.get() {
                        FieldRow {
                            key: field.name.clone(),
                            snapshot: field.clone(),
                            component_name: comp_name.get(),
                        }
                    }
                }
            }
        }
    }
}

/// A single field row — delegates to field_editors which uses prop_controls.
#[component]
fn FieldRow(snapshot: FieldSnapshot, component_name: String) -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd = use_context::<CommandSender>();

    let on_change: Rc<dyn Fn(FieldValue)> = {
        let comp = component_name.clone();
        let fname = snapshot.name.clone();
        let cmd = cmd.clone();
        Rc::new(move |value: FieldValue| {
            if let Some(snap) = store.inspector.get() {
                if let Ok(eid) = uuid::Uuid::parse_str(&snap.entity_id) {
                    let serialized = serde_json::to_string(&value).unwrap_or_default();
                    let _ = cmd.0.send(rkp_engine::EngineCommand::SetComponentField {
                        entity_id: eid,
                        component_name: comp.clone(),
                        field_name: fname.clone(),
                        value: serialized,
                    });
                }
            }
        })
    };

    rsx! {
        {field_editors::field_editor(__scope, &snapshot, on_change)}
    }
}

/// Material usage section — which materials are used and how many voxels of each.
#[component]
fn MaterialUsageSection() -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd_tx: CmdSignal = Signal::new(use_context::<CommandSender>().0);
    let collapsed = Signal::new(false);

    let entity_id = Memo::new(move || {
        store.inspector.get().map(|s| s.entity_id.clone()).unwrap_or_default()
    });

    let usage = Memo::new(move || {
        store.inspector.get()
            .map(|snap| snap.material_usage.clone())
            .unwrap_or_default()
    });

    rsx! {
        if !usage.get().is_empty() {
            div {
                {prop_section_header(__scope, "Materials", collapsed, None)}

                if !collapsed.get() {
                    div {
                        style: "padding:6px 12px;display:flex;flex-direction:column;gap:2px;",
                        for mu in usage.get() {
                            div {
                                key: format!("{}-{}", entity_id.get(), mu.material_id),
                                {material_usage_row(
                                    __scope,
                                    mu.material_id,
                                    mu.voxel_count,
                                    store,
                                    cmd_tx,
                                )}
                            }
                        }
                    }
                }
            }
        }
    }
}

/// A single material usage row with swatch, name, count, and drag-drop remap.
fn material_usage_row(
    __scope: &mut rinch::core::dom::RenderScope,
    material_id: u16,
    voxel_count: u32,
    store: EditorStore,
    cmd_tx: CmdSignal,
) -> rinch::core::dom::NodeHandle {
    let mat_name = Signal::new({
        store.materials.get()
            .iter()
            .find(|m| m.id == material_id)
            .map(|m| m.name.clone())
            .unwrap_or_else(|| format!("Material {material_id}"))
    });
    let mat_color = Signal::new({
        store.materials.get()
            .iter()
            .find(|m| m.id == material_id)
            .map(|m| m.base_color)
            .unwrap_or([0.5, 0.5, 0.5, 1.0])
    });

    let is_drop_target = Signal::new(false);
    let count_str = format_voxel_count(voxel_count);

    rsx! {
        div {
            style: {move || {
                if is_drop_target.get() {
                    "display:flex;align-items:center;gap:6px;padding:3px 4px;\
                     border-radius:3px;border:1px dashed #4fc3f7;background:#1a2a3a;"
                } else {
                    "display:flex;align-items:center;gap:6px;padding:3px 4px;\
                     border-radius:3px;border:1px solid transparent;"
                }
            }},
            ondragenter: move || {
                if store.material_drag.get().is_some() {
                    is_drop_target.set(true);
                }
            },
            ondragleave: move || {
                is_drop_target.set(false);
            },
            ondrop: move || {
                is_drop_target.set(false);
                if let Some(new_mat_id) = store.material_drag.get() {
                    if let Some(snap) = store.inspector.get() {
                        if let Ok(eid) = uuid::Uuid::parse_str(&snap.entity_id) {
                            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::RemapMaterial {
                                object_id: eid,
                                from_material: material_id,
                                to_material: new_mat_id,
                            });
                        }
                    }
                    store.material_drag.set(None);
                }
            },

            // Color swatch
            div {
                style: {move || {
                    let [r, g, b, _] = mat_color.get();
                    format!(
                        "width:14px;height:14px;border-radius:3px;flex-shrink:0;\
                         border:1px solid #3c3c3c;\
                         background:rgb({},{},{});",
                        (r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8,
                    )
                }},
            }

            // Material name
            div {
                style: "flex:1;font-size:11px;color:#ccc;\
                        overflow:hidden;text-overflow:ellipsis;white-space:nowrap;",
                {move || mat_name.get()}
            }

            // Voxel count
            div {
                style: "font-size:10px;color:#666;flex-shrink:0;font-family:monospace;",
                {count_str}
            }
        }
    }
}

fn format_voxel_count(count: u32) -> String {
    if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}K", count as f64 / 1_000.0)
    } else {
        format!("{count}")
    }
}

/// Add Component dropdown button.
#[component]
fn AddComponentButton() -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd_tx: CmdSignal = Signal::new(use_context::<CommandSender>().0);
    let open = Signal::new(false);

    rsx! {
        div {
            style: "padding:8px 12px;",
            {prop_button(__scope, "Add Component", "#2d2d2d", Rc::new(move || {
                open.update(|o| *o = !*o);
            }))}
            // Dropdown
            if open.get() {
                div {
                    style: "margin-top:4px;background:#2d2d2d;border:1px solid #3c3c3c;\
                            border-radius:4px;overflow:hidden;",
                    for name in store.available_components.get() {
                        div {
                            key: name.clone(),
                            style: "padding:6px 12px;cursor:pointer;font-size:11px;color:#ccc;",
                            onclick: {
                                let name = name.clone();
                                move || {
                                    if let Some(snap) = store.inspector.get() {
                                        if let Ok(eid) = uuid::Uuid::parse_str(&snap.entity_id) {
                                            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::AddComponent {
                                                entity_id: eid,
                                                component_name: name.clone(),
                                            });
                                        }
                                    }
                                    open.set(false);
                                }
                            },
                            {name}
                        }
                    }
                }
            }
        }
    }
}
