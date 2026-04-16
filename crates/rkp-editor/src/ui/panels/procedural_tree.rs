//! Procedural node tree widget.
//!
//! Lifted out of `build_panel.rs` so both the right-hand build panel
//! (historical home) and the build viewport's floating overlay can
//! embed the same tree without code duplication. The widget reads a
//! `ProceduralSnapshot` Memo and a `selected_node` Memo; clicks on
//! rows send `SelectProceduralNode`, and the "+" buttons attached to
//! combinators / leaf roots send `AddProceduralNode`. No direct
//! coupling to the surrounding panel's layout — the caller decides
//! padding, scrolling, and sizing.

use rinch::prelude::*;
use rinch_tabler_icons::{TablerIcon, TablerIconStyle, render_tabler_icon};

use rkp_engine::procedural_snapshot::{ProceduralNodeKind, ProceduralSnapshot};

type Scope = rinch::core::dom::RenderScope;
type Node = rinch::core::dom::NodeHandle;

/// Render the whole tree from the snapshot's root. `snapshot` is
/// the full procedural state (nodes, root id, selection); the widget
/// renders only the hierarchy — the caller wraps with whatever chrome
/// (scroll region, background, header) makes sense for its context.
pub fn render_tree(
    __scope: &mut Scope,
    snapshot: Memo<ProceduralSnapshot>,
    selected_node: Memo<Option<u32>>,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> Node {
    // Wrap the root id in a single-element Vec memo so the `for` loop
    // reactively re-renders whenever the snapshot's root changes —
    // auto-promotion (wrap_in_union) moves the root to a new NodeId,
    // and a bare `root_id.get()` in rsx! would not refresh on that
    // change (same pattern as the children loop below).
    let root_ids = Memo::new(move || vec![snapshot.get().root]);

    rsx! {
        div {
            for id in root_ids.get() {
                {render_tree_node(__scope, snapshot, id, 0, selected_node, cmd_tx)}
            }
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
    let is_root_here = Memo::new(move || node_info.get().map(|n| n.is_root).unwrap_or(false));
    let node_kind = Memo::new(move || node_info.get().map(|n| n.kind).unwrap_or(ProceduralNodeKind::Union));

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

                // Add child button. Combinators always show it; a leaf
                // root also shows it — clicking promotes the leaf into
                // a new Union (engine-side auto-promote) so you can
                // attach siblings without manually wrapping first.
                if !is_leaf.get() || is_root_here.get() {
                    {render_add_child_menu(__scope, node_id, node_kind.get(), cmd_tx)}
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

/// Renders the "+" button on combinator rows, opening a popover with a
/// shape/combinator picker. Selected kind is sent as AddProceduralNode.
///
/// `parent_kind` gates context-sensitive entries: `Plane` is an infinite
/// half-space, only useful as a cutter inside Intersect/Subtract — it's
/// hidden from Union children and from leaf roots (which auto-promote
/// to Union on add).
fn render_add_child_menu(
    __scope: &mut Scope,
    parent_id: u32,
    parent_kind: ProceduralNodeKind,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> Node {
    let opened = Signal::new(false);
    let allow_plane = matches!(
        parent_kind,
        ProceduralNodeKind::Intersect | ProceduralNodeKind::Subtract
    );

    rsx! {
        div {
            style: "margin-left:auto;",
            // Stop the row's onclick (node selection) from firing on these events.
            onclick: move || {},

            Popover {
                opened: {move || opened.get()},
                // bottom_end anchors the dropdown's right edge to the +
                // button's right edge so the menu grows leftward. The
                // + sits at the row's right margin; anchoring "bottom"
                // (centered) pushes the menu past the panel edge and
                // clips it.
                position: "bottom_end",
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
                    if allow_plane {
                        {add_menu_item(__scope, "Plane", TablerIcon::LayoutBoard, parent_id, opened, cmd_tx)}
                    }
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
    // Note: `{kind}` (plain variable in expression position) doesn't
    // render as visible text in a `DropdownMenuItem` child — only
    // literal string children or closures take the text-node codegen
    // path. Using `{move || kind}` forces the closure path.
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
            {move || kind}
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
        ProceduralNodeKind::Ramp => TablerIcon::Triangle,
        ProceduralNodeKind::Union => TablerIcon::CirclePlus,
        ProceduralNodeKind::Intersect => TablerIcon::CircleDot,
        ProceduralNodeKind::Subtract => TablerIcon::CircleMinus,
    }
}
