//! Build panel — procedural object node tree editor.
//!
//! Self-contained panel: shows the node tree, param editing for the selected node,
//! and add/remove controls. Only active when the selected entity has a
//! ProceduralGeometry component.

use std::rc::Rc;

use rinch::prelude::*;
use rinch_tabler_icons::{TablerIcon, TablerIconStyle, render_tabler_icon};

use super::prop_controls::*;
use crate::CommandSender;
use crate::ui::store::EditorStore;
use rkp_engine::procedural_snapshot::{
    ProceduralNodeInfo, ProceduralNodeKind, ProceduralParam, ProceduralParamValue,
    ProceduralSnapshot,
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
            // ── Node tree ─────────────────────────────────────────────
            div {
                style: "flex:1;min-height:0;overflow-y:auto;padding:4px;\
                        border-bottom:1px solid #333;",
                {render_tree(__scope, snapshot, selected_node, cmd_tx)}
            }
            // ── Node params ───────────────────────────────────────────
            div {
                style: "flex:1;min-height:0;overflow-y:auto;padding:4px 8px;",
                {render_params(__scope, snapshot, selected_node, cmd_tx)}
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

// ── Node tree ───────────────────────────────────────────────────────────

fn render_tree(
    __scope: &mut Scope,
    snapshot: Memo<ProceduralSnapshot>,
    selected_node: Memo<Option<u32>>,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> Node {
    let root_id = Memo::new(move || snapshot.get().root);

    rsx! {
        div {
            {render_tree_node(__scope, snapshot, root_id.get(), 0, selected_node, cmd_tx)}
        }
    }
}

fn render_tree_node(
    __scope: &mut Scope,
    snapshot: Memo<ProceduralSnapshot>,
    node_id: u32,
    depth: u32,
    selected_node: Memo<Option<u32>>,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> Node {
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
                        style: {move || if collapsed.get() {
                            "font-size:8px;color:#666;cursor:pointer;width:12px;text-align:center;\
                             transform:rotate(-90deg);transition:transform 0.15s;"
                        } else {
                            "font-size:8px;color:#666;cursor:pointer;width:12px;text-align:center;\
                             transition:transform 0.15s;"
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

                // Add child button (only for combinators) — opens a shape picker.
                if !is_leaf.get() {
                    {render_add_child_menu(__scope, node_id, cmd_tx)}
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

    // Position control — prop_vec3 needs a Signal, bridge from snapshot.
    let pos_signal = Signal::new(position.get());
    // Build the position control eagerly (before rsx) to avoid Rc move issues.
    let pos_control = {
        let on_change: Rc<dyn Fn([f32; 3])> = Rc::new(move |val: [f32; 3]| {
            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetProceduralNodePosition {
                node_id: node_id.get(),
                position: glam::Vec3::from(val),
            });
        });
        prop_vec3(__scope, "Position", pos_signal, on_change)
    };

    rsx! {
        div {
            {prop_section_header(__scope, &node_name.get(), collapsed, on_remove)}
            // Position always visible (not inside collapsible)
            div {
                style: "padding:4px 0;",
                {pos_control}
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
            let display = Memo::new(move || signal.get());
            let name = param.name.clone();
            let on_change: Rc<dyn Fn(f32)> = Rc::new(move |val: f32| {
                signal.set(val);
                let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetProceduralNodeParam {
                    node_id: node_id.get(),
                    param_name: name.clone(),
                    value: format!("{val}"),
                });
            });
            prop_scrub(__scope, &param.name, display, min, max, 0.01, on_change)
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

// ── Add-child menu ──────────────────────────────────────────────────────

/// Renders the "+" button on combinator rows, opening a popover with a
/// shape/combinator picker. Selected kind is sent as AddProceduralNode.
fn render_add_child_menu(
    __scope: &mut Scope,
    parent_id: u32,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> Node {
    let opened = Signal::new(false);

    rsx! {
        div {
            style: "margin-left:auto;",
            // Stop the row's onclick (node selection) from firing on these events.
            onclick: move || {},

            Popover {
                opened: {move || opened.get()},
                position: "bottom",
                PopoverTarget {
                    span {
                        style: "color:#666;cursor:pointer;font-size:14px;padding:0 4px;\
                                user-select:none;",
                        onclick: move || opened.update(|v| *v = !*v),
                        "+"
                    }
                }
                PopoverDropdown {
                    {add_menu_item(__scope, "Sphere", TablerIcon::Sphere, parent_id, opened, cmd_tx)}
                    {add_menu_item(__scope, "Box", TablerIcon::Box, parent_id, opened, cmd_tx)}
                    {add_menu_item(__scope, "Capsule", TablerIcon::Capsule, parent_id, opened, cmd_tx)}
                    {add_menu_item(__scope, "Cylinder", TablerIcon::Cylinder, parent_id, opened, cmd_tx)}
                    {add_menu_item(__scope, "Torus", TablerIcon::CircleDotted, parent_id, opened, cmd_tx)}
                    {add_menu_item(__scope, "Plane", TablerIcon::LayoutBoard, parent_id, opened, cmd_tx)}
                    {add_menu_item(__scope, "Ramp", TablerIcon::Triangle, parent_id, opened, cmd_tx)}
                    DropdownMenuDivider {}
                    {add_menu_item(__scope, "Union", TablerIcon::CirclePlus, parent_id, opened, cmd_tx)}
                    {add_menu_item(__scope, "Intersect", TablerIcon::CircleDot, parent_id, opened, cmd_tx)}
                    {add_menu_item(__scope, "Subtract", TablerIcon::CircleMinus, parent_id, opened, cmd_tx)}
                }
            }
        }
    }
}

fn add_menu_item(
    __scope: &mut Scope,
    kind: &'static str,
    icon: TablerIcon,
    parent_id: u32,
    opened: Signal<bool>,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> Node {
    rsx! {
        DropdownMenuItem {
            left_section: icon,
            onclick: move || {
                let _ = cmd_tx.get().send(rkp_engine::EngineCommand::AddProceduralNode {
                    parent_node_id: parent_id,
                    kind: kind.to_string(),
                });
                opened.set(false);
            },
            {kind}
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn node_icon(kind: ProceduralNodeKind) -> TablerIcon {
    match kind {
        ProceduralNodeKind::Sphere => TablerIcon::Sphere,
        ProceduralNodeKind::Box => TablerIcon::Box,
        ProceduralNodeKind::Capsule => TablerIcon::Capsule,
        ProceduralNodeKind::Cylinder => TablerIcon::Cylinder,
        ProceduralNodeKind::Torus => TablerIcon::CircleDotted,
        ProceduralNodeKind::Plane => TablerIcon::LayoutBoard,
        ProceduralNodeKind::Ramp => TablerIcon::Triangle,
        ProceduralNodeKind::Union => TablerIcon::CirclePlus,
        ProceduralNodeKind::Intersect => TablerIcon::CircleDot,
        ProceduralNodeKind::Subtract => TablerIcon::CircleMinus,
    }
}
