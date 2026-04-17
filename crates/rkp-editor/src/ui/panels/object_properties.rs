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

            // Dedicated Animation panel — renders when the entity has a
            // loaded skeleton. Hides the generic Skeleton + AnimationPlayer
            // sections below (they'd just duplicate these controls with
            // uglier ergonomics — text input for clip name etc.).
            if store.inspector.get().as_ref().is_some_and(|s| s.skeleton.is_some()) {
                AnimationSection {
                    key: entity_id.get(),
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
    // Hide Skeleton + AnimationPlayer — the dedicated AnimationSection
    // above renders them more ergonomically (clip dropdown instead of a
    // free-form text input, per-clip scrubber max, bone tree).
    if snapshot.name == "Skeleton" || snapshot.name == "AnimationPlayer" {
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

// ── Animation section (dedicated panel for entities with a Skeleton) ──────

/// Rendered at the top of ObjectProperties for animated entities. Hosts
/// the clip picker, playback transport, and a collapsed bone hierarchy
/// — anything that depends on the loaded `SkeletonAsset`'s contents
/// (clips, bones) rather than the generic component-field reflection.
#[component]
fn AnimationSection() -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd_tx: CmdSignal = Signal::new(use_context::<CommandSender>().0);
    let collapsed = Signal::new(false);
    let bones_collapsed = Signal::new(true);

    let snapshot = Memo::new(move || store.inspector.get());

    rsx! {
        div {
            {prop_section_header(__scope, "Animation", collapsed, None)}

            if !collapsed.get() {
                div {
                    style: "padding:6px 12px;display:flex;flex-direction:column;gap:4px;",
                    {render_animation_body(__scope, snapshot.get(), bones_collapsed, cmd_tx)}
                }
            }
        }
    }
}

/// Render the inner body of the AnimationSection — splits the `snap` /
/// `skel` destructure into a plain function so the rsx macro isn't
/// trying to close over `Option`-moved locals inside its generated
/// effect.
fn render_animation_body(
    __scope: &mut rinch::core::dom::RenderScope,
    snap_opt: Option<InspectorSnapshot>,
    bones_collapsed: Signal<bool>,
    cmd_tx: CmdSignal,
) -> rinch::core::dom::NodeHandle {
    let Some(snap) = snap_opt else { return rsx! { span {} }; };
    let Some(skel) = snap.skeleton.clone() else { return rsx! { span {} }; };
    let controls = animation_controls(__scope, snap, skel.clone(), cmd_tx);
    let tree = bone_tree(__scope, skel, bones_collapsed);
    rsx! {
        div {
            style: "display:flex;flex-direction:column;gap:4px;",
            {controls}
            {tree}
        }
    }
}

fn animation_controls(
    __scope: &mut rinch::core::dom::RenderScope,
    snap: InspectorSnapshot,
    skel: SkeletonInspector,
    cmd_tx: CmdSignal,
) -> rinch::core::dom::NodeHandle {
    // Read AnimationPlayer state from the component snapshots.
    let clip_name = match player_field(&snap.components, "clip_name") {
        Some(FieldValue::String(s)) => s.clone(),
        _ => String::new(),
    };
    let time = match player_field(&snap.components, "time") {
        Some(FieldValue::Float(v)) => *v as f32,
        _ => 0.0,
    };
    let speed = match player_field(&snap.components, "speed") {
        Some(FieldValue::Float(v)) => *v as f32,
        _ => 1.0,
    };
    let playing = match player_field(&snap.components, "playing") {
        Some(FieldValue::Bool(b)) => *b,
        _ => false,
    };
    let loop_mode = match player_field(&snap.components, "loop_mode") {
        Some(FieldValue::String(s)) => s.clone(),
        _ => "Loop".into(),
    };

    // Clip dropdown options — built from the loaded asset's clip list.
    // Empty fallback shows "(no clip)" as the single option.
    let clip_options_owned: Vec<(String, String)> = if skel.clips.is_empty() {
        vec![(String::new(), "(no clip)".into())]
    } else {
        skel.clips.iter().map(|c| (c.name.clone(), format!("{} ({:.2}s)", c.name, c.duration))).collect()
    };
    let clip_options_refs: Vec<(&str, &str)> = clip_options_owned.iter()
        .map(|(v, l)| (v.as_str(), l.as_str())).collect();

    // Clip duration drives the scrubber range; fall back to 1s when
    // unknown so the slider never has max<min.
    let duration = skel.clips.iter()
        .find(|c| c.name == clip_name)
        .map(|c| c.duration)
        .filter(|d| *d > 0.0)
        .unwrap_or(1.0);

    // `Signal` is `Copy` — closures below just deref it each call,
    // so a single binding captured by value into each `Rc<Fn>` is fine.
    let entity_id = Signal::new(snap.entity_id.clone());

    let clip_signal = Signal::new(clip_name.clone());
    let time_signal = Signal::new(time);
    let speed_signal = Signal::new(speed);
    let loop_signal = Signal::new(loop_mode.clone());

    let on_clip = Rc::new(move |v: String| {
        send_field_edit(cmd_tx.get(), &entity_id.get(), "AnimationPlayer", "clip_name", FieldValue::String(v));
    }) as Rc<dyn Fn(String)>;
    let on_time = Rc::new(move |v: f32| {
        send_field_edit(cmd_tx.get(), &entity_id.get(), "AnimationPlayer", "time", FieldValue::Float(v as f64));
    }) as Rc<dyn Fn(f32)>;
    let on_speed = Rc::new(move |v: f32| {
        send_field_edit(cmd_tx.get(), &entity_id.get(), "AnimationPlayer", "speed", FieldValue::Float(v as f64));
    }) as Rc<dyn Fn(f32)>;
    let on_loop = Rc::new(move |v: String| {
        send_field_edit(cmd_tx.get(), &entity_id.get(), "AnimationPlayer", "loop_mode", FieldValue::String(v));
    }) as Rc<dyn Fn(String)>;

    let play_accent = if playing { "#2e5a2e" } else { "#2d2d2d" };
    let on_play = Rc::new(move || {
        send_field_edit(cmd_tx.get(), &entity_id.get(), "AnimationPlayer", "playing", FieldValue::Bool(!playing));
    }) as Rc<dyn Fn()>;
    let on_stop = Rc::new(move || {
        send_field_edit(cmd_tx.get(), &entity_id.get(), "AnimationPlayer", "playing", FieldValue::Bool(false));
        send_field_edit(cmd_tx.get(), &entity_id.get(), "AnimationPlayer", "time", FieldValue::Float(0.0));
    }) as Rc<dyn Fn()>;

    let loop_options: [(&str, &str); 3] = [
        ("Once", "Once"),
        ("Loop", "Loop"),
        ("PingPong", "PingPong"),
    ];

    // Master skinning toggle — when `false`, the engine skips the
    // scatter pass and the march shader renders the asset rigidly at
    // rest pose. Defaults on.
    let store_for_skin = use_context::<EditorStore>();
    let skinning_signal = store_for_skin.skinning_enabled;
    let dqs_signal = store_for_skin.dqs_enabled;
    let cmd_tx_skin = cmd_tx.clone();
    let on_skinning = Rc::new(move |enabled: bool| {
        let _ = cmd_tx_skin.get().send(rkp_engine::EngineCommand::SetViewOption {
            option: "skinning".into(),
            enabled,
        });
    }) as Rc<dyn Fn(bool)>;
    let on_dqs = Rc::new(move |enabled: bool| {
        let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetViewOption {
            option: "dqs".into(),
            enabled,
        });
    }) as Rc<dyn Fn(bool)>;

    let path_sig = Signal::new(skel.path.clone());
    rsx! {
        div {
            style: "display:flex;flex-direction:column;gap:2px;",
            // Path (read-only).
            {prop_label(__scope, "Asset", path_sig)}
            // Clip dropdown.
            {prop_select(__scope, "Clip", clip_signal, &clip_options_refs, on_clip)}
            // Transport row — Play/Pause and Stop side-by-side.
            div {
                style: "display:flex;gap:6px;padding:4px 0;",
                div {
                    style: "flex:1;",
                    {prop_button(__scope, if playing { "Pause" } else { "Play" }, play_accent, on_play)}
                }
                div {
                    style: "flex:1;",
                    {prop_button(__scope, "Stop", "#2d2d2d", on_stop)}
                }
            }
            // Time scrubber (range = clip duration).
            {prop_scrub(__scope, "Time", time_signal, 0.0, duration, 0.01, on_time)}
            // Speed scrub.
            {prop_scrub(__scope, "Speed", speed_signal, -4.0, 4.0, 0.01, on_speed)}
            // Loop mode.
            {prop_select(__scope, "Loop", loop_signal, &loop_options, on_loop)}
            // Master skinning toggle (off = rigid render).
            {prop_checkbox(__scope, "Skinning", skinning_signal, on_skinning)}
            // LBS vs DQS. On = DQS (preserves joint volume).
            {prop_checkbox(__scope, "DQS", dqs_signal, on_dqs)}
        }
    }
}

fn bone_tree(
    __scope: &mut rinch::core::dom::RenderScope,
    skel: SkeletonInspector,
    collapsed: Signal<bool>,
) -> rinch::core::dom::NodeHandle {
    let bone_count = skel.bone_names.len();
    let title = format!("Bones ({bone_count})");
    // Precompute rows once and stash in a Signal so the rsx `for`
    // generated closure can `.get()` a fresh clone each render without
    // moving out of a captured local.
    let rows: Signal<Vec<BoneRow>> = Signal::new(render_bone_rows(&skel));

    rsx! {
        div {
            style: "margin-top:6px;border-top:1px solid #3c3c3c;padding-top:4px;",
            {prop_section_header(__scope, &title, collapsed, None)}

            if !collapsed.get() {
                div {
                    style: "padding:4px 6px;display:flex;flex-direction:column;gap:1px;\
                            font-size:11px;font-family:monospace;color:#aaa;\
                            max-height:240px;overflow-y:auto;",
                    for row in rows.get() {
                        div {
                            key: format!("bone-{}", row.index),
                            style: "white-space:pre;",
                            {row.display}
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone, PartialEq)]
struct BoneRow {
    index: usize,
    display: String,
}

/// Indent bone names by hierarchy depth. Children always come after
/// parents (glTF/FBX both guarantee this in the extractor), so a single
/// linear pass that counts ancestor hops is enough.
fn render_bone_rows(skel: &SkeletonInspector) -> Vec<BoneRow> {
    let mut depths = vec![0u32; skel.bone_names.len()];
    for i in 0..skel.bone_names.len() {
        let parent = skel.bone_parents.get(i).copied().unwrap_or(-1);
        if parent >= 0 {
            let p = parent as usize;
            if p < i {
                depths[i] = depths[p] + 1;
            }
        }
    }
    skel.bone_names.iter().enumerate().map(|(i, name)| {
        let indent = "  ".repeat(depths[i] as usize);
        BoneRow {
            index: i,
            display: format!("{indent}{name}"),
        }
    }).collect()
}
