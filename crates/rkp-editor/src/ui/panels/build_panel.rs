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
            // ── Resolution ────────────────────────────────────────────
            {render_resolution(__scope, snapshot, cmd_tx)}
            // ── Node params ───────────────────────────────────────────
            // Tree widget and Preview/Bake controls both live on the
            // build viewport as floating overlays; this panel is just
            // resolution + the currently-selected node's parameters.
            div {
                style: "flex:1;min-height:0;overflow-y:auto;padding:4px 8px;",
                {render_params(__scope, store.clone(), snapshot, selected_node, cmd_tx)}
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
    store: EditorStore,
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
                {render_node_params(__scope, store.clone(), node_info, cmd_tx)}
            }
        }
    }
}

fn render_node_params(
    __scope: &mut Scope,
    store: EditorStore,
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
                        {render_param_field(__scope, store.clone(), node_id, param.clone(), cmd_tx)}
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
        ProceduralParamValue::Color(v) => {
            let signal = Signal::new(v);
            let name = param.name.clone();
            // Engine-side parser is `parse_vec3` → "r,g,b", so drop
            // alpha in the wire format. Color params today are pure RGB
            // under the hood (glam::Vec3); the picker carries alpha for
            // symmetry with material colors but nobody reads it.
            let on_change: Rc<dyn Fn([f32; 4])> = Rc::new(move |val: [f32; 4]| {
                let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetProceduralNodeParam {
                    node_id: node_id.get(),
                    param_name: name.clone(),
                    value: format!("{},{},{}", val[0], val[1], val[2]),
                });
            });
            prop_color(__scope, &param.name, signal, on_change)
        }
        ProceduralParamValue::Material(mat_id) => {
            material_slot_row(__scope, store, node_id, &param.name, mat_id, cmd_tx)
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
    mat_id: u16,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> Node {
    let param_name_owned = param_name.to_string();
    let label_text = param_name.to_string();

    // Current material name + swatch color, looked up against the
    // scene's material library. Memos so a material-palette edit or
    // tombstone refreshes the display without a param round-trip.
    let mat_name = Memo::new(move || {
        store.materials.get()
            .iter()
            .find(|m| m.id == mat_id)
            .map(|m| m.name.clone())
            .unwrap_or_else(|| format!("Material {mat_id}"))
    });
    let mat_color = Memo::new(move || {
        store.materials.get()
            .iter()
            .find(|m| m.id == mat_id)
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

