//! Build panel — preview controls + parameter editor for the selected
//! procedural node.
//!
//! The node tree itself lives in `procedural_tree.rs` and is embedded
//! by the build viewport as a floating overlay; this panel is now
//! specifically the right-hand companion showing Preview toggle, Bake
//! button, Resolution picker, and parameter fields for whatever node
//! the tree has selected. Only active when the selected entity has a
//! ProceduralGeometry component.

use std::rc::Rc;

use rinch::prelude::*;

use super::prop_controls::*;
use crate::CommandSender;
use crate::ui::store::EditorStore;
use rkp_engine::procedural_snapshot::{
    ProceduralNodeInfo, ProceduralParam, ProceduralParamValue, ProceduralSnapshot,
};

type Scope = rinch::core::dom::RenderScope;
type Node = rinch::core::dom::NodeHandle;

#[component]
pub fn BuildPanel() -> NodeHandle {
    let store = use_context::<EditorStore>();

    rsx! {
        div {
            style: "display:flex;flex-direction:column;height:100%;overflow-y:auto;\
                    color:#ccc;font-size:12px;",
            if store.procedural.get().is_some() {
                {build_content(__scope, store)}
            }
            if store.procedural.get().is_none() {
                div {
                    style: "display:flex;align-items:center;justify-content:center;\
                            height:100%;color:#666;font-style:italic;",
                    "Select a procedural object"
                }
            }
        }
    }
}

fn build_content(__scope: &mut Scope, store: EditorStore) -> Node {
    let cmd_tx = Signal::new(use_context::<CommandSender>().0);

    let snapshot = Memo::new(move || store.procedural.get().unwrap_or_default());
    let selected_node = Memo::new(move || snapshot.get().selected_node);

    rsx! {
        div {
            style: "display:flex;flex-direction:column;height:100%;",
            // ── Preview mode toggle (voxel vs live CSG raymarch) ──────
            {render_preview_toggle(__scope, store, cmd_tx)}
            // ── Bake action ───────────────────────────────────────────
            {render_bake_action(__scope, snapshot, cmd_tx)}
            // ── Resolution ────────────────────────────────────────────
            {render_resolution(__scope, snapshot, cmd_tx)}
            // ── Node params ───────────────────────────────────────────
            // The tree widget lives on the build viewport as a floating
            // overlay now; this panel is just preview controls + the
            // currently-selected node's parameters.
            div {
                style: "flex:1;min-height:0;overflow-y:auto;padding:4px 8px;",
                {render_params(__scope, snapshot, selected_node, cmd_tx)}
            }
        }
    }
}

/// Two-button segmented control for the build viewport's primary-
/// visibility source. `Voxel` shows the baked octree result — the
/// same thing the main viewport sees; becomes stale the moment the
/// user edits the tree. `Raymarch` evaluates the tree analytically
/// per pixel, so it's always in sync with the current parameters but
/// doesn't reflect material/lighting quality exactly. The pair is
/// the expected editing loop: edit with Raymarch, Bake, confirm with
/// Voxel.
fn render_preview_toggle(
    __scope: &mut Scope,
    store: EditorStore,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> Node {
    // Mirror the engine's current preview mode in a local signal so the
    // buttons re-highlight when the engine reports a change. This also
    // gives us an easy place to flip the mode without touching the
    // engine on every render.
    let mode = store.build_preview_mode;

    let set_voxel = move || {
        mode.set(rkp_render::BuildPreviewMode::Voxel);
        let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetBuildPreviewMode {
            mode: rkp_render::BuildPreviewMode::Voxel,
        });
    };
    let set_raymarch = move || {
        mode.set(rkp_render::BuildPreviewMode::Raymarch);
        let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetBuildPreviewMode {
            mode: rkp_render::BuildPreviewMode::Raymarch,
        });
    };

    let btn_style = |active: bool| -> &'static str {
        if active {
            "flex:1;padding:4px 8px;background:#2a4e7a;color:#fff;\
             border:1px solid #3a6ea6;cursor:pointer;"
        } else {
            "flex:1;padding:4px 8px;background:#2a2a2a;color:#aaa;\
             border:1px solid #333;cursor:pointer;"
        }
    };

    rsx! {
        div {
            style: "display:flex;align-items:center;gap:8px;padding:6px 8px;\
                    border-bottom:1px solid #333;",
            span { style: "color:#888;font-size:11px;", "Preview:" }
            div {
                style: "display:flex;flex:1;gap:0;",
                button {
                    style: {move || btn_style(matches!(mode.get(), rkp_render::BuildPreviewMode::Voxel))},
                    onclick: set_voxel,
                    "Voxel"
                }
                button {
                    style: {move || btn_style(matches!(mode.get(), rkp_render::BuildPreviewMode::Raymarch))},
                    onclick: set_raymarch,
                    "Live (raymarch)"
                }
            }
        }
    }
}

/// Bake button + dirty indicator. Interactive edits mark the tree dirty
/// but don't rebake — the user clicks this to pay the voxelization cost
/// on demand. The button highlights when there are unbaked changes.
fn render_bake_action(
    __scope: &mut Scope,
    snapshot: Memo<ProceduralSnapshot>,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> Node {
    let dirty = Memo::new(move || snapshot.get().dirty);
    let entity_id = Memo::new(move || snapshot.get().entity_id);

    let on_click = move || {
        let _ = cmd_tx.get().send(rkp_engine::EngineCommand::BakeProceduralEntity {
            entity_id: entity_id.get(),
        });
    };

    rsx! {
        div {
            style: "display:flex;align-items:center;gap:8px;padding:6px 8px;\
                    border-bottom:1px solid #333;",
            button {
                // Highlighted blue when dirty, neutral gray when clean. Clicking
                // when clean is a no-op cost-wise (full rebake runs anyway) so
                // don't disable — just de-emphasize.
                style: {move || if dirty.get() {
                    "flex:1;padding:6px 10px;background:#2a4e7a;color:#fff;\
                     border:1px solid #3a6ea6;border-radius:3px;cursor:pointer;\
                     font-weight:600;"
                } else {
                    "flex:1;padding:6px 10px;background:#2a2a2a;color:#888;\
                     border:1px solid #444;border-radius:3px;cursor:pointer;"
                }},
                onclick: on_click,
                "Bake"
            }
            span {
                style: {move || if dirty.get() {
                    "color:#f0a04b;font-size:11px;"
                } else {
                    "color:#666;font-size:11px;"
                }},
                {move || if dirty.get() { "unbaked changes" } else { "up to date" }}
            }
        }
    }
}

// ── Resolution control ──────────────────────────────────────────────────

fn render_resolution(
    __scope: &mut Scope,
    snapshot: Memo<ProceduralSnapshot>,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> Node {
    let vs = snapshot.get().voxel_size;
    let current = Signal::new(format!("{vs}"));
    let on_change: Rc<dyn Fn(String)> = Rc::new(move |v: String| {
        let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetProceduralVoxelSize {
            tier: v,
        });
    });

    rsx! {
        div {
            style: "padding:4px 8px;border-bottom:1px solid #333;",
            {prop_select(
                __scope,
                "Resolution",
                current,
                &[
                    ("0.005", "5mm (finest)"),
                    ("0.02", "2cm"),
                    ("0.08", "8cm"),
                    ("0.32", "32cm (coarsest)"),
                ],
                on_change,
            )}
        }
    }
}

// ── Node params ─────────────────────────────────────────────────────────

fn render_params(
    __scope: &mut Scope,
    snapshot: Memo<ProceduralSnapshot>,
    selected_node: Memo<Option<u32>>,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> Node {
    let node_info = Memo::new(move || {
        let snap = snapshot.get();
        let sel = selected_node.get()?;
        snap.nodes.iter().find(|n| n.id == sel).cloned()
    });

    rsx! {
        div {
            if node_info.get().is_none() {
                div {
                    style: "color:#666;font-style:italic;padding:8px;",
                    "Select a node to edit"
                }
            }
            if node_info.get().is_some() {
                {render_node_params(__scope, node_info, cmd_tx)}
            }
        }
    }
}

fn render_node_params(
    __scope: &mut Scope,
    node_info: Memo<Option<ProceduralNodeInfo>>,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> Node {
    let node_id = Memo::new(move || node_info.get().map(|n| n.id).unwrap_or(0));
    let node_name = Memo::new(move || node_info.get().map(|n| n.name.clone()).unwrap_or_default());
    let params = Memo::new(move || node_info.get().map(|n| n.params.clone()).unwrap_or_default());
    let is_root = Memo::new(move || node_info.get().map(|n| n.is_root).unwrap_or(true));
    let position = Memo::new(move || node_info.get().map(|n| n.position).unwrap_or([0.0; 3]));
    let rotation = Memo::new(move || node_info.get().map(|n| n.rotation).unwrap_or([0.0; 3]));
    let scale = Memo::new(move || node_info.get().map(|n| n.scale).unwrap_or([1.0; 3]));

    let collapsed = Signal::new(false);

    let on_remove: Option<Rc<dyn Fn()>> = if !is_root.get() {
        Some(Rc::new(move || {
            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::RemoveProceduralNode {
                node_id: node_id.get(),
            });
        }))
    } else {
        None
    };

    // Transform controls. Each `prop_vec3` wants a Signal; we bridge
    // from the snapshot Memos. The signals are local to this render —
    // prop_vec3 reads the initial value on mount and only fires
    // on_change on user edit, so missing back-propagation from the
    // Memo is not a correctness issue for text-entry/drag interactions.
    let pos_signal = Signal::new(position.get());
    let rot_signal = Signal::new(rotation.get());
    let scale_signal = Signal::new(scale.get());

    let pos_control = {
        let on_change: Rc<dyn Fn([f32; 3])> = Rc::new(move |val: [f32; 3]| {
            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetProceduralNodePosition {
                node_id: node_id.get(),
                position: glam::Vec3::from(val),
            });
        });
        prop_vec3(__scope, "Position", pos_signal, on_change)
    };
    let rot_control = {
        let on_change: Rc<dyn Fn([f32; 3])> = Rc::new(move |val: [f32; 3]| {
            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetProceduralNodeRotation {
                node_id: node_id.get(),
                rotation_deg: glam::Vec3::from(val),
            });
        });
        prop_vec3(__scope, "Rotation", rot_signal, on_change)
    };
    let scale_control = {
        let on_change: Rc<dyn Fn([f32; 3])> = Rc::new(move |val: [f32; 3]| {
            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetProceduralNodeScale {
                node_id: node_id.get(),
                scale: glam::Vec3::from(val),
            });
        });
        prop_vec3(__scope, "Scale", scale_signal, on_change)
    };

    rsx! {
        div {
            {prop_section_header(__scope, &node_name.get(), collapsed, on_remove)}
            // Transform block — always visible (not inside the collapsible
            // params section) since it's the primary handle for any node.
            div {
                style: "padding:4px 0;display:flex;flex-direction:column;gap:2px;",
                {pos_control}
                {rot_control}
                {scale_control}
            }
            if !collapsed.get() {
                div {
                    style: "display:flex;flex-direction:column;gap:2px;",
                    for param in params.get() {
                        {render_param_field(__scope, node_id, param.clone(), cmd_tx)}
                    }
                }
            }
        }
    }
}

fn render_param_field(
    __scope: &mut Scope,
    node_id: Memo<u32>,
    param: ProceduralParam,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> Node {
    match param.value {
        ProceduralParamValue::Float(v) => {
            let (min, max) = param.range.unwrap_or((0.0, 100.0));
            let signal = Signal::new(v);
            let name = param.name.clone();
            let on_change: Rc<dyn Fn(f32)> = Rc::new(move |val: f32| {
                let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetProceduralNodeParam {
                    node_id: node_id.get(),
                    param_name: name.clone(),
                    value: format!("{val}"),
                });
            });
            prop_scrub(__scope, &param.name, signal, min, max, 0.01, on_change)
        }
        ProceduralParamValue::Vec3(v) => {
            let signal = Signal::new(v);
            let name = param.name.clone();
            let on_change: Rc<dyn Fn([f32; 3])> = Rc::new(move |val: [f32; 3]| {
                let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetProceduralNodeParam {
                    node_id: node_id.get(),
                    param_name: name.clone(),
                    value: format!("{},{},{}", val[0], val[1], val[2]),
                });
            });
            prop_vec3(__scope, &param.name, signal, on_change)
        }
        ProceduralParamValue::U16(v) => {
            let signal = Signal::new(v as i64);
            let name = param.name.clone();
            let on_change: Rc<dyn Fn(i64)> = Rc::new(move |val: i64| {
                let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetProceduralNodeParam {
                    node_id: node_id.get(),
                    param_name: name.clone(),
                    value: format!("{val}"),
                });
            });
            prop_number_i64(__scope, &param.name, signal, on_change)
        }
        ProceduralParamValue::MaterialCombine(ref v) => {
            let signal = Signal::new(v.clone());
            let name = param.name.clone();
            let on_change: Rc<dyn Fn(String)> = Rc::new(move |val: String| {
                let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetProceduralNodeParam {
                    node_id: node_id.get(),
                    param_name: name.clone(),
                    value: val,
                });
            });
            prop_select(
                __scope,
                &param.name,
                signal,
                &[("Winner", "Winner"), ("Layered", "Layered"), ("Blend", "Blend")],
                on_change,
            )
        }
    }
}

