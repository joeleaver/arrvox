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

mod materials;
mod skeleton;

use materials::MaterialUsageSection;
use skeleton::skeleton_component_section;

/// Send a single `SetComponentField` edit to the engine for the selected entity.
fn send_field_edit(
    cmd_tx: crossbeam::channel::Sender<rkp_engine::EngineCommand>,
    entity_id: &str,
    component: &str,
    field: &str,
    value: FieldValue,
) {
    let Ok(eid) = uuid::Uuid::parse_str(entity_id) else { return };
    let serialized = serde_json::to_string(&value).unwrap_or_default();
    let _ = cmd_tx.send(rkp_engine::EngineCommand::SetComponentField {
        entity_id: eid,
        component_name: component.into(),
        field_name: field.into(),
        value: serialized,
    });
}

/// Pull a current field value off the AnimationPlayer component snapshot.
fn player_field<'a>(
    components: &'a [ComponentSnapshot],
    field: &str,
) -> Option<&'a FieldValue> {
    components.iter()
        .find(|c| c.name == "AnimationPlayer")?
        .fields.iter()
        .find(|f| f.name == field)
        .map(|f| &f.value)
}

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

/// Structural-only key for the component for-loop. Does NOT include
/// field values — that's the whole point. The for-loop's reconciler
/// re-runs ComponentSection only when components are added/removed
/// or the selection changes, NOT every time a Transform's position
/// updates. Per-field reactivity happens inside ComponentSection /
/// FieldRow via Memos that read from `store.inspector` directly and
/// short-circuit on PartialEq.
#[derive(Clone, PartialEq)]
struct KeyedComponent {
    entity_id: String,
    component_name: String,
    removable: bool,
}

fn flatten_keyed(keyed: (String, Vec<(String, bool)>)) -> Vec<KeyedComponent> {
    let (entity_id, comps) = keyed;
    comps
        .into_iter()
        .map(|(component_name, removable)| KeyedComponent {
            entity_id: entity_id.clone(),
            component_name,
            removable,
        })
        .collect()
}

#[component]
fn InspectorContent() -> NodeHandle {
    let store = use_context::<EditorStore>();

    // Memo emits ONLY the structural keys: the entity id and the list of
    // component (name, removable) pairs. Field values are intentionally
    // excluded — when only `Transform.position` changes, this memo's
    // value is unchanged (PartialEq), so rinch's for-loop reconciler
    // doesn't tear down ComponentSection. Per-field reactive updates
    // happen one level deeper, via Memos that each FieldRow derives
    // against `store.inspector`.
    //
    // Combining entity_id + components into a single memo also keeps the
    // for-loop effect subscribed to one source — separate memos used to
    // re-queue the loop effect mid-flush, panicking on the trailing
    // RefCell borrow.
    let keyed_components = Memo::new(move || {
        store.inspector.get()
            .map(|snap| (
                snap.entity_id.clone(),
                snap.components.iter()
                    .map(|c| (c.name.clone(), c.removable))
                    .collect::<Vec<_>>(),
            ))
            .unwrap_or_default()
    });

    let entity_name = Memo::new(move || {
        store.inspector.get().map(|s| s.entity_name.clone()).unwrap_or_default()
    });

    let entity_id = Memo::new(move || {
        store.inspector.get().map(|s| s.entity_id.clone()).unwrap_or_default()
    });

    // Procedural-only badge in the header. Reads the same
    // `store.procedural` snapshot that drives the build viewport's
    // "unbaked / up to date" indicator, so both surfaces always agree.
    let proc_unbaked = Memo::new(move || {
        store.procedural.get().is_some_and(|s| s.dirty)
    });
    let proc_present = Memo::new(move || store.procedural.get().is_some());

    rsx! {
        div {
            style: "display:flex;flex-direction:column;",

            // Entity header
            div {
                style: "padding:8px 12px;background:#2d2d2d;border-bottom:1px solid #3c3c3c;",
                div {
                    style: "display:flex;align-items:center;gap:8px;",
                    div {
                        style: "font-weight:600;font-size:13px;flex:1;\
                                overflow:hidden;text-overflow:ellipsis;white-space:nowrap;",
                        {move || entity_name.get()}
                    }
                    if proc_present.get() {
                        span {
                            style: {move || if proc_unbaked.get() {
                                "font-size:10px;color:#f0a04b;\
                                 padding:1px 6px;border:1px solid #6a4520;\
                                 border-radius:8px;background:#2a1f15;white-space:nowrap;"
                            } else {
                                "font-size:10px;color:#666;\
                                 padding:1px 6px;border:1px solid #3a3a3a;\
                                 border-radius:8px;background:#252525;white-space:nowrap;"
                            }},
                            {move || if proc_unbaked.get() { "● unbaked" } else { "● baked" }}
                        }
                    }
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
                    key: format!("{}-{}", keyed.entity_id, keyed.component_name),
                    entity_id: keyed.entity_id,
                    component_name: keyed.component_name,
                    removable: keyed.removable,
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
///
/// Takes only `entity_id` + `component_name` + `removable` as props (the
/// structural keys), NOT the snapshot itself. The component data is
/// re-read reactively via Memos against `store.inspector`. Result: when
/// only a field VALUE changes (e.g. physics writing Transform.position),
/// this component is preserved and only the affected field's DOM
/// updates via the per-field Memo inside `FieldRow`.
#[component]
fn ComponentSection(entity_id: String, component_name: String, removable: bool) -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd_tx: CmdSignal = Signal::new(use_context::<CommandSender>().0);
    let collapsed = Signal::new(false);

    // Hide EditorMetadata — it's shown in the header.
    if component_name == "EditorMetadata" {
        return rsx! { span {} };
    }
    // AnimationPlayer is bundled into the Skeleton section below —
    // hide its own entry in the list so transport lives in one place.
    if component_name == "AnimationPlayer" {
        return rsx! { span {} };
    }
    // Skeleton gets a rich custom body instead of generic field
    // reflection — clip dropdown, transport, scrubber, DQS toggle,
    // bone tree. The skeleton helper still wants the full snapshot, so
    // we pull the latest from the store at this point. Skeleton state
    // doesn't churn at 60Hz so the broader-grained reactivity here is
    // fine; if it ever does, refactor the same way as the generic path.
    if component_name == "Skeleton" {
        let snap = store.inspector.get()
            .and_then(|s| s.components.iter().find(|c| c.name == "Skeleton").cloned())
            .unwrap_or_default();
        return skeleton_component_section(__scope, snap);
    }

    // Reactive list of field rows. The Memo computes each field's STATIC
    // metadata bundled with the entity/component identifiers (so the
    // for-body's `Fn` closure doesn't have to borrow them from outer
    // scope — it owns each item). Only changes when the component's
    // schema changes, so the for-loop reconciler preserves existing
    // FieldRows across value updates.
    let cn_for_fields = component_name.clone();
    let entity_id_for_fields = entity_id.clone();
    let field_rows: Memo<Vec<FieldRowProps>> = Memo::new(move || {
        let cn = cn_for_fields.clone();
        let eid = entity_id_for_fields.clone();
        store.inspector.get()
            .and_then(|s| s.components.iter().find(|c| c.name == cn).cloned())
            .map(|c| c.fields.iter().map(|f| FieldRowProps {
                entity_id: eid.clone(),
                component_name: cn.clone(),
                meta: FieldSnapshot {
                    // Strip the live value — flows through a per-field
                    // Memo inside FieldRow. Keeping value here would
                    // refire the for-loop on every value change.
                    name: f.name.clone(),
                    field_type: f.field_type,
                    value: FieldValue::default(),
                    range: f.range,
                    transient: f.transient,
                    asset_filter: f.asset_filter.clone(),
                    enum_options: f.enum_options.clone(),
                    scrub: f.scrub,
                },
            }).collect())
            .unwrap_or_default()
    });

    // Build the remove callback for non-mandatory components.
    let entity_id_for_remove = entity_id.clone();
    let comp_name_for_remove = component_name.clone();
    let on_remove = if removable {
        Some(Rc::new(move || {
            if let Ok(eid) = uuid::Uuid::parse_str(&entity_id_for_remove) {
                let _ = cmd_tx.get().send(rkp_engine::EngineCommand::RemoveComponent {
                    entity_id: eid,
                    component_name: comp_name_for_remove.clone(),
                });
            }
        }) as Rc<dyn Fn()>)
    } else {
        None
    };

    rsx! {
        div {
            {prop_section_header(__scope, &component_name, collapsed, on_remove)}

            // Fields (hidden when collapsed)
            if !collapsed.get() {
                div {
                    style: "padding:6px 12px;display:flex;flex-direction:column;gap:2px;",
                    for row in field_rows.get() {
                        FieldRow {
                            key: row.meta.name.clone(),
                            entity_id: row.entity_id,
                            component_name: row.component_name,
                            meta: row.meta,
                        }
                    }
                }
            }
        }
    }
}

/// Per-field-row payload: structural keys + static metadata. Used as
/// the for-loop item type so the loop body's `Fn` closure owns each
/// item rather than borrowing from outer scope.
#[derive(Clone, PartialEq)]
struct FieldRowProps {
    entity_id: String,
    component_name: String,
    meta: FieldSnapshot,
}

/// A single field row. Reads its value reactively via a per-field Memo
/// against `store.inspector` — when only this field changes, only this
/// row's DOM updates; sibling fields stay completely quiet.
#[component]
fn FieldRow(entity_id: String, component_name: String, meta: FieldSnapshot) -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd = use_context::<CommandSender>();

    // Per-field value Memo — short-circuits on PartialEq so unchanged
    // fields never push DOM updates.
    let cn_for_value = component_name.clone();
    let fname_for_value = meta.name.clone();
    let value: Memo<FieldValue> = Memo::new(move || {
        store.inspector.get()
            .and_then(|s| s.components.iter().find(|c| c.name == cn_for_value).cloned())
            .and_then(|c| c.fields.iter().find(|f| f.name == fname_for_value).cloned())
            .map(|f| f.value)
            .unwrap_or_default()
    });

    let comp = component_name;
    let fname = meta.name.clone();
    let on_change: Rc<dyn Fn(FieldValue)> = Rc::new(move |new_value: FieldValue| {
        if let Ok(eid) = uuid::Uuid::parse_str(&entity_id) {
            let serialized = serde_json::to_string(&new_value).unwrap_or_default();
            let _ = cmd.0.send(rkp_engine::EngineCommand::SetComponentField {
                entity_id: eid,
                component_name: comp.clone(),
                field_name: fname.clone(),
                value: serialized,
            });
        }
    });

    rsx! {
        {field_editors::field_editor(__scope, &meta, value, on_change)}
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

// ── Skeleton component section — custom body with rich animation UI ──────

