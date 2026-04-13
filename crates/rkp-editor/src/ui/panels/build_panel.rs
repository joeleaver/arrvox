//! Build panel — procedural object node tree editor.
//!
//! Self-contained panel: shows the node tree, param editing for the selected node,
//! and add/remove controls. Only active when the selected entity has a
//! ProceduralGeometry component.

use rinch::prelude::*;
use rinch_tabler_icons::{TablerIcon, TablerIconStyle, render_tabler_icon};

use crate::CommandSender;
use crate::ui::store::EditorStore;
use rkp_engine::procedural_snapshot::{
    ProceduralNodeInfo, ProceduralNodeKind, ProceduralParam, ProceduralParamValue, ProceduralSnapshot,
};

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

fn build_content(
    __scope: &mut rinch::core::dom::RenderScope,
    store: EditorStore,
) -> rinch::core::dom::NodeHandle {
    let cmd_tx = Signal::new(use_context::<CommandSender>().0);

    let snapshot = Memo::new(move || store.procedural.get().unwrap_or_default());
    let selected_node = Memo::new(move || snapshot.get().selected_node);

    rsx! {
        div {
            style: "display:flex;flex-direction:column;height:100%;",
            // ── Node tree ─────────────────────────────────────────────
            div {
                style: "flex:1;min-height:0;overflow-y:auto;padding:4px;\
                        border-bottom:1px solid #333;",
                // Header
                div {
                    style: "display:flex;align-items:center;justify-content:space-between;\
                            padding:2px 4px 6px;",
                    span { style: "font-weight:600;color:#999;text-transform:uppercase;\
                                   font-size:10px;letter-spacing:0.5px;",
                        "Node Tree"
                    }
                }
                // Tree
                {render_tree(__scope, snapshot, selected_node, cmd_tx)}
            }
            // ── Node params ───────────────────────────────────────────
            div {
                style: "flex:1;min-height:0;overflow-y:auto;padding:4px;",
                {render_params(__scope, snapshot, selected_node, cmd_tx)}
            }
        }
    }
}

fn render_tree(
    __scope: &mut rinch::core::dom::RenderScope,
    snapshot: Memo<ProceduralSnapshot>,
    selected_node: Memo<Option<u32>>,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> rinch::core::dom::NodeHandle {
    let root_id = Memo::new(move || snapshot.get().root);

    rsx! {
        div {
            {render_tree_node(__scope, snapshot, root_id.get(), 0, selected_node, cmd_tx)}
        }
    }
}

fn render_tree_node(
    __scope: &mut rinch::core::dom::RenderScope,
    snapshot: Memo<ProceduralSnapshot>,
    node_id: u32,
    depth: u32,
    selected_node: Memo<Option<u32>>,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> rinch::core::dom::NodeHandle {
    let indent = depth as f32 * 16.0;
    let collapsed = Signal::new(false);

    let node_info = Memo::new(move || {
        snapshot.get().nodes.iter().find(|n| n.id == node_id).cloned()
    });

    let icon = Memo::new(move || {
        node_info.get().map(|n| node_icon(n.kind)).unwrap_or(TablerIcon::Cube)
    });

    let name = Memo::new(move || {
        node_info.get().map(|n| n.name.clone()).unwrap_or_default()
    });

    let children = Memo::new(move || {
        node_info.get().map(|n| n.children.clone()).unwrap_or_default()
    });

    let has_children = Memo::new(move || !children.get().is_empty());
    let is_leaf = Memo::new(move || node_info.get().map(|n| n.is_leaf).unwrap_or(true));

    rsx! {
        div {
            // Row
            div {
                style: {move || {
                    let sel = selected_node.get() == Some(node_id);
                    let bg = if sel { "background:#37373d;color:#fff;" } else { "color:#ccc;" };
                    format!(
                        "display:flex;align-items:center;padding:2px 8px 2px {:.0}px;\
                         cursor:pointer;border-radius:3px;font-size:12px;gap:4px;{bg}",
                        8.0 + indent
                    )
                }},
                onclick: move || {
                    let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SelectProceduralNode {
                        node_id: Some(node_id),
                    });
                },

                // Chevron
                if has_children.get() {
                    span {
                        style: {move || {
                            if collapsed.get() {
                                "font-size:8px;color:#666;cursor:pointer;width:12px;text-align:center;\
                                 transform:rotate(-90deg);transition:transform 0.15s;"
                            } else {
                                "font-size:8px;color:#666;cursor:pointer;width:12px;text-align:center;\
                                 transition:transform 0.15s;"
                            }
                        }},
                        onclick: move || collapsed.update(|c| *c = !*c),
                        {"\u{25BC}"}
                    }
                }
                if !has_children.get() {
                    span { style: "width:12px;" }
                }

                // Icon
                span {
                    style: "width:14px;height:14px;display:inline-flex;align-items:center;\
                            justify-content:center;flex-shrink:0;color:#999;",
                    {render_tabler_icon(__scope, icon.get(), TablerIconStyle::Outline)}
                }

                // Name
                span {
                    style: "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;",
                    {|| name.get()}
                }

                // Add child button (only for combinators)
                if !is_leaf.get() {
                    span {
                        style: "margin-left:auto;color:#666;cursor:pointer;font-size:14px;\
                                padding:0 2px;",
                        onclick: move || {
                            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::AddProceduralNode {
                                parent_node_id: node_id,
                                kind: "Sphere".to_string(),
                            });
                        },
                        "+"
                    }
                }
            }

            // Children
            if !collapsed.get() {
                for child_id in children.get() {
                    {render_tree_node(__scope, snapshot, child_id, depth + 1, selected_node, cmd_tx)}
                }
            }
        }
    }
}

fn render_params(
    __scope: &mut rinch::core::dom::RenderScope,
    snapshot: Memo<ProceduralSnapshot>,
    selected_node: Memo<Option<u32>>,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> rinch::core::dom::NodeHandle {
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
    __scope: &mut rinch::core::dom::RenderScope,
    node_info: Memo<Option<ProceduralNodeInfo>>,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> rinch::core::dom::NodeHandle {
    let node_id = Memo::new(move || node_info.get().map(|n| n.id).unwrap_or(0));
    let node_name = Memo::new(move || node_info.get().map(|n| n.name.clone()).unwrap_or_default());
    let params = Memo::new(move || node_info.get().map(|n| n.params.clone()).unwrap_or_default());
    let is_root = Memo::new(move || {
        // Can't delete root — node_id 0 is always root.
        node_info.get().map(|n| n.id == 0).unwrap_or(true)
    });

    rsx! {
        div {
            // Header with node name + delete button
            div {
                style: "display:flex;align-items:center;justify-content:space-between;\
                        padding:2px 4px 6px;",
                span {
                    style: "font-weight:600;color:#999;text-transform:uppercase;\
                            font-size:10px;letter-spacing:0.5px;",
                    {|| node_name.get()}
                }
                if !is_root.get() {
                    span {
                        style: "color:#e55;cursor:pointer;font-size:11px;padding:2px 4px;\
                                border-radius:3px;",
                        onclick: move || {
                            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::RemoveProceduralNode {
                                node_id: node_id.get(),
                            });
                        },
                        "Delete"
                    }
                }
            }

            // Parameter fields
            for param in params.get() {
                {render_param_field(__scope, node_id, param.clone(), cmd_tx)}
            }
        }
    }
}

fn render_param_field(
    __scope: &mut rinch::core::dom::RenderScope,
    node_id: Memo<u32>,
    param: ProceduralParam,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> rinch::core::dom::NodeHandle {
    let display_name = param.name.clone();

    // Pre-extract from param.value before rsx to avoid moves.
    let editor = match param.value {
        ProceduralParamValue::Float(v) => {
            render_float_field(__scope, node_id, param.name.clone(), v, param.range, cmd_tx)
        }
        ProceduralParamValue::Vec3(v) => {
            render_vec3_field(__scope, node_id, param.name.clone(), v, cmd_tx)
        }
        ProceduralParamValue::U16(v) => {
            render_u16_field(__scope, node_id, param.name.clone(), v, cmd_tx)
        }
        ProceduralParamValue::MaterialCombine(v) => {
            render_enum_field(__scope, node_id, param.name.clone(), v.clone(), cmd_tx)
        }
    };

    rsx! {
        div {
            style: "display:flex;align-items:center;padding:2px 4px;gap:8px;",
            // Label
            span {
                style: "width:90px;flex-shrink:0;color:#999;font-size:11px;",
                {display_name}
            }
            // Value editor
            {editor}
        }
    }
}

fn render_float_field(
    __scope: &mut rinch::core::dom::RenderScope,
    node_id: Memo<u32>,
    param_name: String,
    value: f32,
    _range: Option<(f32, f32)>,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> rinch::core::dom::NodeHandle {
    let local = Signal::new(value);

    rsx! {
        input {
            r#type: "number",
            style: "flex:1;background:#2a2a2a;border:1px solid #444;border-radius:3px;\
                    color:#ccc;padding:2px 4px;font-size:11px;font-family:monospace;\
                    min-width:0;",
            step: "0.01",
            value: {move || format!("{:.3}", local.get())},
            oninput: {
                let name = param_name.clone();
                move |val: String| {
                    if let Ok(f) = val.parse::<f32>() {
                        local.set(f);
                        let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetProceduralNodeParam {
                            node_id: node_id.get(),
                            param_name: name.clone(),
                            value: val,
                        });
                    }
                }
            },
        }
    }
}

fn render_vec3_field(
    __scope: &mut rinch::core::dom::RenderScope,
    node_id: Memo<u32>,
    param_name: String,
    value: [f32; 3],
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> rinch::core::dom::NodeHandle {
    let text = Signal::new(format!("{:.3},{:.3},{:.3}", value[0], value[1], value[2]));

    rsx! {
        input {
            style: "flex:1;background:#2a2a2a;border:1px solid #444;border-radius:3px;\
                    color:#ccc;padding:2px 4px;font-size:11px;font-family:monospace;\
                    min-width:0;",
            value: {move || text.get()},
            oninput: {
                let name = param_name.clone();
                move |val: String| {
                    text.set(val.clone());
                    let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetProceduralNodeParam {
                        node_id: node_id.get(),
                        param_name: name.clone(),
                        value: val,
                    });
                }
            },
        }
    }
}

fn render_u16_field(
    __scope: &mut rinch::core::dom::RenderScope,
    node_id: Memo<u32>,
    param_name: String,
    value: u16,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> rinch::core::dom::NodeHandle {
    let local = Signal::new(value as i64);

    rsx! {
        input {
            r#type: "number",
            style: "flex:1;background:#2a2a2a;border:1px solid #444;border-radius:3px;\
                    color:#ccc;padding:2px 4px;font-size:11px;font-family:monospace;\
                    min-width:0;",
            step: "1",
            value: {move || local.get().to_string()},
            oninput: {
                let name = param_name.clone();
                move |val: String| {
                    if let Ok(i) = val.parse::<i64>() {
                        local.set(i);
                        let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetProceduralNodeParam {
                            node_id: node_id.get(),
                            param_name: name.clone(),
                            value: val,
                        });
                    }
                }
            },
        }
    }
}

fn render_enum_field(
    __scope: &mut rinch::core::dom::RenderScope,
    node_id: Memo<u32>,
    param_name: String,
    value: String,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> rinch::core::dom::NodeHandle {
    rsx! {
        div {
            style: "display:flex;gap:2px;",
            {render_enum_option(__scope, node_id, param_name.clone(), "Winner".into(), value.clone(), cmd_tx)}
            {render_enum_option(__scope, node_id, param_name.clone(), "Layered".into(), value.clone(), cmd_tx)}
            {render_enum_option(__scope, node_id, param_name.clone(), "Blend".into(), value.clone(), cmd_tx)}
        }
    }
}

fn render_enum_option(
    __scope: &mut rinch::core::dom::RenderScope,
    node_id: Memo<u32>,
    param_name: String,
    option: String,
    current: String,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> rinch::core::dom::NodeHandle {
    let is_active = option == current;
    let style = if is_active {
        "background:#37373d;color:#fff;padding:2px 6px;border-radius:3px;font-size:11px;\
         cursor:pointer;border:1px solid #555;"
    } else {
        "background:#2a2a2a;color:#999;padding:2px 6px;border-radius:3px;font-size:11px;\
         cursor:pointer;border:1px solid #333;"
    };

    rsx! {
        span {
            style: style,
            onclick: {
                let name = param_name.clone();
                let val = option.clone();
                move || {
                    let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetProceduralNodeParam {
                        node_id: node_id.get(),
                        param_name: name.clone(),
                        value: val.clone(),
                    });
                }
            },
            {option}
        }
    }
}

fn node_icon(kind: ProceduralNodeKind) -> TablerIcon {
    match kind {
        ProceduralNodeKind::Sphere => TablerIcon::Sphere,
        ProceduralNodeKind::Box => TablerIcon::Box,
        ProceduralNodeKind::Capsule => TablerIcon::Capsule,
        ProceduralNodeKind::Cylinder => TablerIcon::Cylinder,
        ProceduralNodeKind::Torus => TablerIcon::CircleDotted,
        ProceduralNodeKind::Plane => TablerIcon::LayoutBoard,
        ProceduralNodeKind::Union => TablerIcon::CirclePlus,
        ProceduralNodeKind::Intersect => TablerIcon::CircleDot,
        ProceduralNodeKind::Subtract => TablerIcon::CircleMinus,
    }
}
