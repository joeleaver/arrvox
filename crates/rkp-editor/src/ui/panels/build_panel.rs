//! Build panel — parameter editor + resolution picker for the selected
//! procedural node.
//!
//! The node tree and the Preview / Bake controls both live as floating
//! overlays on the build viewport now (see `procedural_tree.rs` and
//! `build_viewport.rs`). This panel is the right-hand companion showing
//! the Resolution picker and the parameter fields for whatever node the
//! tree has selected. Only active when the selected entity has a
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
            // The tree, preview toggle, bake button, resolution picker,
            // and voxel-count readout all live as floating overlays on
            // the build viewport — this panel is now only the selected
            // node's parameters.
            div {
                style: "flex:1;min-height:0;overflow-y:auto;padding:4px 8px;",
                {render_params(__scope, store.clone(), snapshot, selected_node, cmd_tx)}
            }
        }
    }
}

// ── Node params ─────────────────────────────────────────────────────────
//
// All local Signals bound to the selected node's state (transform
// fields, param values, title) are created once when the component
// mounts. That's correct as long as we REMOUNT the section whenever
// the selected node changes — otherwise the signals stay pinned to
// the previous selection and the UI shows stale title + stale field
// values. We get remount-on-selection by iterating a 1-element Vec
// keyed on the node id: when the id changes the previous subtree
// unmounts and a fresh `NodeParamsSection` mounts. Same pattern
// `object_properties.rs` uses for its ComponentSection, and the one
// called out in `feedback_rinch_ui.md` as the canonical fix for
// stale-form-on-selection.

fn render_params(
    __scope: &mut Scope,
    _store: EditorStore,
    snapshot: Memo<ProceduralSnapshot>,
    selected_node: Memo<Option<u32>>,
    _cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> Node {
    let node_info = Memo::new(move || {
        let snap = snapshot.get();
        let sel = selected_node.get()?;
        snap.nodes.iter().find(|n| n.id == sel).cloned()
    });

    // Wrap in a Vec so we can drive a keyed `for` loop — the Vec has
    // 0 or 1 items; changing the selected node id changes the key,
    // which unmounts the old section and mounts a fresh one.
    let selected_list = Memo::new(move || {
        node_info.get().into_iter().collect::<Vec<ProceduralNodeInfo>>()
    });

    rsx! {
        div {
            if node_info.get().is_none() {
                div {
                    style: "color:#666;font-style:italic;padding:8px;",
                    "Select a node to edit"
                }
            }
            for info in selected_list.get() {
                NodeParamsSection {
                    key: info.id.to_string(),
                    node_info: info,
                }
            }
        }
    }
}

#[component]
fn NodeParamsSection(node_info: ProceduralNodeInfo) -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>> =
        Signal::new(use_context::<CommandSender>().0);

    // node_id and is_root are stable for a given selection — captured
    // from the mount-time snapshot. Everything else is read from the
    // live snapshot via a Memo below, so external updates flow in.
    let node_id = node_info.id;
    let node_id_memo = Memo::new(move || node_id);
    let is_root = node_info.is_root;

    // Every dynamic field (name, transform components, the per-param
    // list) is read reactively from `store.procedural` via a Memo.
    // When the engine pushes a new snapshot (viewport drag, MCP,
    // undo/redo, material-by-height param change, etc.), the Memo
    // re-fires and the control reflects it — without any signal-sync
    // Effect. Writes still go through `cmd_tx` so the engine stays
    // authoritative.
    let lookup_node = move || -> Option<ProceduralNodeInfo> {
        store.procedural.get()?
            .nodes
            .iter()
            .find(|n| n.id == node_id)
            .cloned()
    };
    let pos_memo = Memo::new(move || lookup_node().map(|n| n.position).unwrap_or([0.0; 3]));
    let rot_memo = Memo::new(move || lookup_node().map(|n| n.rotation).unwrap_or([0.0; 3]));
    let scale_memo = Memo::new(move || lookup_node().map(|n| n.scale).unwrap_or([1.0; 3]));
    let params_memo = Memo::new(move || lookup_node().map(|n| n.params).unwrap_or_default());
    let name_memo = Memo::new(move || lookup_node().map(|n| n.name).unwrap_or_default());

    let collapsed = Signal::new(false);

    let on_remove: Option<Rc<dyn Fn()>> = if !is_root {
        Some(Rc::new(move || {
            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::RemoveProceduralNode {
                node_id,
            });
        }))
    } else {
        None
    };

    let pos_control = {
        let on_change: Rc<dyn Fn([f32; 3])> = Rc::new(move |val: [f32; 3]| {
            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetProceduralNodePosition {
                node_id,
                position: glam::Vec3::from(val),
            });
        });
        prop_vec3(__scope, "Position", pos_memo, on_change)
    };
    let rot_control = {
        let on_change: Rc<dyn Fn([f32; 3])> = Rc::new(move |val: [f32; 3]| {
            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetProceduralNodeRotation {
                node_id,
                rotation_deg: glam::Vec3::from(val),
            });
        });
        prop_vec3(__scope, "Rotation", rot_memo, on_change)
    };
    let scale_control = {
        let on_change: Rc<dyn Fn([f32; 3])> = Rc::new(move |val: [f32; 3]| {
            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetProceduralNodeScale {
                node_id,
                scale: glam::Vec3::from(val),
            });
        });
        prop_vec3(__scope, "Scale", scale_memo, on_change)
    };

    // Inline the section header here so the title can read from
    // `name_memo` reactively — `prop_section_header` takes a `&str`
    // and is used by many other panels with static titles, so instead
    // of changing its signature we replicate its layout once (it's
    // just the chevron + title + optional remove button).
    let removable = on_remove.is_some();
    let on_remove_sig: Signal<Option<Rc<dyn Fn()>>> = Signal::new(on_remove);

    rsx! {
        div {
            // Header.
            div {
                style: "display:flex;align-items:center;padding:6px 12px;cursor:pointer;\
                        background:#2a2a2a;gap:6px;border-bottom:1px solid #3c3c3c;",
                onclick: move || collapsed.update(|c| *c = !*c),
                span {
                    style: {move || {
                        if collapsed.get() {
                            "font-size:10px;color:#666;transform:rotate(-90deg);\
                             transition:transform 0.15s;display:inline-block;"
                        } else {
                            "font-size:10px;color:#666;transition:transform 0.15s;\
                             display:inline-block;"
                        }
                    }},
                    {"\u{25BC}"}
                }
                span {
                    style: "flex:1;font-size:11px;font-weight:600;color:#bbb;\
                            text-transform:uppercase;letter-spacing:0.3px;",
                    {move || name_memo.get()}
                }
                if removable {
                    div {
                        style: "cursor:pointer;color:#666;width:14px;height:14px;\
                                display:flex;align-items:center;justify-content:center;\
                                border-radius:2px;flex-shrink:0;",
                        onclick: move || {
                            if let Some(ref cb) = on_remove_sig.get() {
                                cb();
                            }
                        },
                        {"\u{2715}"}
                    }
                }
            }
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
                    for param in params_memo.get() {
                        {render_param_field(__scope, store.clone(), node_id_memo, param, cmd_tx)}
                    }
                }
            }
        }
    }
}

fn render_param_field(
    __scope: &mut Scope,
    store: EditorStore,
    node_id: Memo<u32>,
    param: ProceduralParam,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> Node {
    // Each param's current value is looked up reactively from the
    // snapshot (by node id + param name) instead of being captured
    // one-shot from `param.value` at mount — so an external change
    // (viewport drag, MCP, undo) flowing through the snapshot shows
    // up in the field without a rebuild. `param.value` still carries
    // the param's TYPE and bounds (the discriminant at mount time
    // tells us which control to render); the numeric / string /
    // color payload is re-read from the Memo each fire.
    let name_for_memo = param.name.clone();
    let lookup_param = move || -> Option<ProceduralParamValue> {
        let snap = store.procedural.get()?;
        let nid = node_id.get();
        let n = snap.nodes.iter().find(|n| n.id == nid)?;
        n.params.iter().find(|p| p.name == name_for_memo).map(|p| p.value.clone())
    };

    match param.value {
        ProceduralParamValue::Float(v) => {
            let (min, max) = param.range.unwrap_or((0.0, 100.0));
            let name = param.name.clone();
            let value_memo = Memo::new(move || match lookup_param() {
                Some(ProceduralParamValue::Float(f)) => f,
                _ => v,
            });
            let on_change: Rc<dyn Fn(f32)> = Rc::new(move |val: f32| {
                let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetProceduralNodeParam {
                    node_id: node_id.get(),
                    param_name: name.clone(),
                    value: format!("{val}"),
                });
            });
            prop_scrub(__scope, &param.name, value_memo, min, max, 0.01, on_change)
        }
        ProceduralParamValue::Color(v) => {
            let default = v;
            let value_memo = Memo::new(move || match lookup_param() {
                Some(ProceduralParamValue::Color(c)) => c,
                _ => default,
            });
            let name = param.name.clone();
            let on_change: Rc<dyn Fn([f32; 4])> = Rc::new(move |val: [f32; 4]| {
                let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetProceduralNodeParam {
                    node_id: node_id.get(),
                    param_name: name.clone(),
                    value: format!("{},{},{}", val[0], val[1], val[2]),
                });
            });
            prop_color(__scope, &param.name, value_memo, on_change)
        }
        ProceduralParamValue::Material(_mat_id) => {
            let mat_memo = Memo::new(move || match lookup_param() {
                Some(ProceduralParamValue::Material(m)) => m,
                _ => _mat_id,
            });
            material_slot_row(__scope, store, node_id, &param.name, mat_memo, cmd_tx)
        }
        ProceduralParamValue::MaterialCombine(ref v) => {
            let default = v.clone();
            let value_memo = Memo::new(move || match lookup_param() {
                Some(ProceduralParamValue::MaterialCombine(s)) => s,
                _ => default.clone(),
            });
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
                value_memo,
                &[("Winner", "Winner"), ("Layered", "Layered"), ("Blend", "Blend")],
                on_change,
            )
        }
        ProceduralParamValue::Select { ref value, ref options } => {
            let default = value.clone();
            let value_memo = Memo::new(move || match lookup_param() {
                Some(ProceduralParamValue::Select { value, .. }) => value,
                _ => default.clone(),
            });
            let name = param.name.clone();
            let on_change: Rc<dyn Fn(String)> = Rc::new(move |val: String| {
                let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetProceduralNodeParam {
                    node_id: node_id.get(),
                    param_name: name.clone(),
                    value: val,
                });
            });
            let opt_refs: Vec<(&str, &str)> = options
                .iter()
                .map(|(v, l)| (v.as_str(), l.as_str()))
                .collect();
            prop_select(__scope, &param.name, value_memo, &opt_refs, on_change)
        }
    }
}

/// Material slot: swatch + name, accepts a drag from the materials panel
/// (via `store.material_drag`). Drop fires a SetProceduralNodeParam with
/// the dropped material id. Mirrors the drop pattern in
/// `object_properties::material_usage_row` so the two panels feel the
/// same to the user.
fn material_slot_row(
    __scope: &mut Scope,
    store: EditorStore,
    node_id: Memo<u32>,
    param_name: &str,
    mat_id: Memo<u16>,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> Node {
    let param_name_owned = param_name.to_string();
    let label_text = param_name.to_string();

    // Current material name + swatch color, looked up against the
    // scene's material library. Memos so a palette edit OR a param
    // change that flips this slot to a new material (viewport drag,
    // MCP, undo) refreshes the display.
    let mat_name = Memo::new(move || {
        let id = mat_id.get();
        store.materials.get()
            .iter()
            .find(|m| m.id == id)
            .map(|m| m.name.clone())
            .unwrap_or_else(|| format!("Material {id}"))
    });
    let mat_color = Memo::new(move || {
        let id = mat_id.get();
        store.materials.get()
            .iter()
            .find(|m| m.id == id)
            .map(|m| m.base_color)
            .unwrap_or([0.5, 0.5, 0.5, 1.0])
    });

    let is_drop_target = Signal::new(false);

    rsx! {
        div {
            style: "display:flex;align-items:center;gap:6px;min-height:22px;",
            div { style: "width:72px;flex-shrink:0;font-size:11px;color:#999;\
                          overflow:hidden;text-overflow:ellipsis;white-space:nowrap;",
                {label_text}
            }
            div {
                style: {move || {
                    if is_drop_target.get() {
                        "flex:1;min-width:0;display:flex;align-items:center;gap:6px;\
                         padding:3px 4px;border-radius:3px;\
                         border:1px dashed #4fc3f7;background:#1a2a3a;"
                    } else {
                        "flex:1;min-width:0;display:flex;align-items:center;gap:6px;\
                         padding:3px 4px;border-radius:3px;\
                         border:1px solid #3c3c3c;background:#1e1e1e;"
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
                        let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetProceduralNodeParam {
                            node_id: node_id.get(),
                            param_name: param_name_owned.clone(),
                            value: format!("{new_mat_id}"),
                        });
                        store.material_drag.set(None);
                    }
                },

                // Color swatch from material base_color.
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
                // Material name.
                div {
                    style: "flex:1;font-size:11px;color:#ccc;\
                            overflow:hidden;text-overflow:ellipsis;white-space:nowrap;",
                    {move || mat_name.get()}
                }
            }
        }
    }
}

