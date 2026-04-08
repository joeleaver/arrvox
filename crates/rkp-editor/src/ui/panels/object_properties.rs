//! Object properties panel — the heart of the editor.
//!
//! Displays and edits all components on the selected entity using the
//! component registry's reflection system. Components are shown as
//! collapsible sections with per-type field editors.

use std::rc::Rc;

use rinch::prelude::*;
use rinch_tabler_icons::{TablerIcon, TablerIconStyle, render_tabler_icon};

use crate::CommandSender;
use crate::ui::store::EditorStore;
use rkp_engine::inspector::*;

use super::field_editors;

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

#[component]
fn InspectorContent() -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd = use_context::<CommandSender>();

    let components = Memo::new(move || {
        store.inspector.get()
            .map(|snap| snap.components.clone())
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

            // Component sections
            for comp in components.get() {
                ComponentSection {
                    key: comp.name.clone(),
                    snapshot: comp.clone(),
                }
            }

            // Add Component button
            AddComponentButton {}
        }
    }
}

/// A single component section — collapsible header + field editors.
#[component]
fn ComponentSection(snapshot: ComponentSnapshot) -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd = use_context::<CommandSender>();
    let collapsed = Signal::new(false);
    let comp_name = Signal::new(snapshot.name.clone());
    let fields = Signal::new(snapshot.fields.clone());
    let removable = snapshot.removable;

    // Hide EditorMetadata — it's shown in the header.
    if snapshot.name == "EditorMetadata" {
        return rsx! { span {} };
    }

    rsx! {
        div {
            style: "border-bottom:1px solid #3c3c3c;",

            // Section header
            div {
                style: "display:flex;align-items:center;padding:6px 12px;cursor:pointer;\
                        background:#2a2a2a;gap:6px;",
                onclick: move || collapsed.update(|c| *c = !*c),

                // Collapse chevron
                span {
                    style: {move || {
                        if collapsed.get() {
                            "font-size:10px;color:#666;transform:rotate(-90deg);transition:transform 0.15s;"
                        } else {
                            "font-size:10px;color:#666;transition:transform 0.15s;"
                        }
                    }},
                    {"\u{25BC}"} // ▼
                }

                // Component name
                span {
                    style: "flex:1;font-size:11px;font-weight:600;color:#bbb;\
                            text-transform:uppercase;letter-spacing:0.3px;",
                    {move || comp_name.get()}
                }

                // Remove button (only for non-mandatory components)
                if removable {
                    div {
                        style: "cursor:pointer;color:#666;width:14px;height:14px;\
                                display:flex;align-items:center;justify-content:center;",
                        onclick: {
                            let cmd = cmd.clone();
                            move || {
                                let cn = comp_name.get();
                                if let Some(snap) = store.inspector.get() {
                                    if let Ok(eid) = uuid::Uuid::parse_str(&snap.entity_id) {
                                        let _ = cmd.0.send(rkp_engine::EngineCommand::RemoveComponent {
                                            entity_id: eid,
                                            component_name: cn,
                                        });
                                    }
                                }
                            }
                        },
                        {render_tabler_icon(__scope, TablerIcon::X, TablerIconStyle::Outline)}
                    }
                }
            }

            // Fields (hidden when collapsed)
            if !collapsed.get() {
                div {
                    style: "padding:6px 12px;",
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

/// A single field row — label + type-appropriate editor.
#[component]
fn FieldRow(snapshot: FieldSnapshot, component_name: String) -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd = use_context::<CommandSender>();
    let field_name = snapshot.name.clone();
    let is_transient = snapshot.transient;

    // Callback for when the field value changes.
    let on_change: Rc<dyn Fn(FieldValue)> = {
        let comp = component_name.clone();
        let fname = field_name.clone();
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
        div {
            style: "display:flex;align-items:center;margin-bottom:4px;gap:4px;",
            // Label
            div {
                style: {|| {
                    if is_transient {
                        "width:80px;flex-shrink:0;color:#555;font-size:11px;font-style:italic;"
                    } else {
                        "width:80px;flex-shrink:0;color:#888;font-size:11px;"
                    }
                }},
                {field_name}
            }
            // Editor
            div {
                style: "flex:1;min-width:0;",
                {field_editors::field_editor(__scope, &snapshot, on_change)}
            }
        }
    }
}

/// Add Component dropdown button.
#[component]
fn AddComponentButton() -> NodeHandle {
    let store = use_context::<EditorStore>();
    // Store the sender in a Signal so it's Copy and can be used in for-loop closures.
    let cmd_tx = Signal::new(use_context::<CommandSender>().0);
    let open = Signal::new(false);

    rsx! {
        div {
            style: "padding:8px 12px;",
            // Button
            div {
                style: "display:flex;align-items:center;justify-content:center;gap:4px;\
                        padding:6px 12px;background:#2d2d2d;border:1px solid #3c3c3c;\
                        border-radius:4px;cursor:pointer;color:#888;font-size:11px;",
                onclick: move || open.update(|o| *o = !*o),
                span {
                    style: "width:12px;height:12px;display:inline-flex;\
                            align-items:center;justify-content:center;",
                    {render_tabler_icon(__scope, TablerIcon::Plus, TablerIconStyle::Outline)}
                }
                {"Add Component"}
            }
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
