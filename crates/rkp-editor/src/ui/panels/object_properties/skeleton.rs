//! Skeleton-component section: bone tree, animation transport, scrubber, skinning toggles.

use std::rc::Rc;

use rinch::prelude::*;

use crate::CommandSender;
use crate::ui::store::EditorStore;
use crate::ui::panels::prop_controls::*;
use rkp_engine::inspector::*;

use super::{CmdSignal, player_field, send_field_edit};

/// Body for a `Skeleton` ComponentSection. Renders the clip picker,
/// playback transport, scrubber, skinning / DQS toggles, and a
/// collapsible bone tree in place of the plain field-reflection list
/// that every other component gets.
pub(super) fn skeleton_component_section(
    __scope: &mut rinch::core::dom::RenderScope,
    snapshot: ComponentSnapshot,
) -> rinch::core::dom::NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd_tx: CmdSignal = Signal::new(use_context::<CommandSender>().0);
    let collapsed = Signal::new(false);
    let bones_collapsed = Signal::new(true);
    let comp_name = Signal::new(snapshot.name.clone());
    let removable = snapshot.removable;

    // Remove callback — firing this drops both Skeleton *and*
    // AnimationPlayer (bundled) via the engine's RemoveComponent
    // handler.
    let on_remove: Option<Rc<dyn Fn()>> = if removable {
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

    let snap_memo = Memo::new(move || store.inspector.get());

    rsx! {
        div {
            {prop_section_header(__scope, "Skeleton", collapsed, on_remove)}

            if !collapsed.get() {
                div {
                    style: "padding:6px 12px;display:flex;flex-direction:column;gap:4px;",
                    {render_skeleton_body(__scope, snap_memo.get(), bones_collapsed, cmd_tx)}
                }
            }
        }
    }
}

/// Render the body of the Skeleton section — splits the `snap`/`skel`
/// destructure into a plain function so the rsx macro isn't trying to
/// close over `Option`-moved locals inside its generated effect.
fn render_skeleton_body(
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
    let loop_mode = match player_field(&snap.components, "loop_mode") {
        Some(FieldValue::String(s)) => s.clone(),
        _ => "Loop".into(),
    };

    // Reactive reads off the inspector — the engine advances `time`
    // every frame and flips `playing` on pause, so the transport
    // controls have to follow inspector state rather than a value
    // captured at render time. Same story for `speed` if another
    // system ever writes to it (e.g. slow-mo).
    let store_react = use_context::<EditorStore>();
    let playing_memo = Memo::new(move || {
        store_react.inspector.get().as_ref()
            .and_then(|s| player_field(&s.components, "playing").cloned())
            .map(|v| matches!(v, FieldValue::Bool(true)))
            .unwrap_or(false)
    });
    let time_memo = Memo::new(move || {
        store_react.inspector.get().as_ref()
            .and_then(|s| player_field(&s.components, "time").cloned())
            .and_then(|v| if let FieldValue::Float(f) = v { Some(f as f32) } else { None })
            .unwrap_or(0.0)
    });
    let speed_memo = Memo::new(move || {
        store_react.inspector.get().as_ref()
            .and_then(|s| player_field(&s.components, "speed").cloned())
            .and_then(|v| if let FieldValue::Float(f) = v { Some(f as f32) } else { None })
            .unwrap_or(1.0)
    });

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

    let on_play = Rc::new(move || {
        let current = playing_memo.get();
        send_field_edit(cmd_tx.get(), &entity_id.get(), "AnimationPlayer", "playing", FieldValue::Bool(!current));
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
            {prop_select(__scope, "Clip", Memo::new(move || clip_signal.get()), &clip_options_refs, on_clip)}
            // Transport row — Play/Pause and Stop side-by-side.
            // Play/Pause is rendered inline so its label and accent
            // update reactively with `playing_memo` (prop_button takes
            // static &str — can't track the signal).
            div {
                style: "display:flex;gap:6px;padding:4px 0;",
                div {
                    style: "flex:1;",
                    div {
                        style: {move || format!(
                            "display:flex;align-items:center;justify-content:center;gap:4px;\
                             padding:6px 12px;background:{c};border:1px solid {c};\
                             border-radius:4px;cursor:pointer;color:#ddd;font-size:11px;\
                             font-weight:500;",
                            c = if playing_memo.get() { "#2e5a2e" } else { "#2d2d2d" },
                        )},
                        onclick: move || on_play(),
                        {move || if playing_memo.get() { "Pause" } else { "Play" }}
                    }
                }
                div {
                    style: "flex:1;",
                    {prop_button(__scope, "Stop", "#2d2d2d", on_stop)}
                }
            }
            // Time scrubber (range = clip duration).
            {prop_scrub(__scope, "Time", time_memo, 0.0, duration, 0.01, on_time)}
            // Speed scrub.
            {prop_scrub(__scope, "Speed", speed_memo, -4.0, 4.0, 0.01, on_speed)}
            // Loop mode.
            {prop_select(__scope, "Loop", Memo::new(move || loop_signal.get()), &loop_options, on_loop)}
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
