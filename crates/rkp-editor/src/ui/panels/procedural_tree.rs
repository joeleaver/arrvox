//! Procedural node tree widget.
//!
//! Lifted out of `build_panel.rs` so both the right-hand build panel
//! (historical home) and the build viewport's floating overlay can
//! embed the same tree without code duplication. The widget reads a
//! `ProceduralSnapshot` Memo and a `selected_node` Memo; clicks on
//! rows send `SelectProceduralNode`, and the "+" buttons attached to
//! combinators / leaf roots send `AddProceduralNode`.
//!
//! Rows are also drag-and-drop sources and targets (reorder siblings
//! or reparent to a combinator) and expose a right-click context
//! menu with Duplicate / Delete actions. The drag context is a
//! single `DragContext<u32>` (the dragged node id) shared across all
//! rows in one render.

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

    // One drag context for the whole tree. `DragContext::new` wraps a
    // Signal<Option<u32>> so it's Copy and can be moved into every row
    // handler without cloning.
    let drag = DragContext::<u32>::new();

    rsx! {
        div {
            for id in root_ids.get() {
                {render_tree_node(__scope, snapshot, id, 0, None, 0, selected_node, cmd_tx, drag)}
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn render_tree_node(
    __scope: &mut Scope,
    snapshot: Memo<ProceduralSnapshot>,
    node_id: u32,
    depth: u32,
    parent_id: Option<u32>,
    sibling_index: u32,
    selected_node: Memo<Option<u32>>,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
    drag: DragContext<u32>,
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
    let is_root = Memo::new(move || node_info.get().map(|n| n.is_root).unwrap_or(false));
    let node_kind = Memo::new(move || node_info.get().map(|n| n.kind).unwrap_or(ProceduralNodeKind::Union));

    // Strips show only while a drag is in progress. Otherwise they're
    // 0-height and don't steal vertical space or hit-testing.
    let drag_active = Memo::new(move || drag.is_active());

    // Local per-row hover state for visual feedback. Using Signals (not
    // Cells) so the conditional `style:` closures pick up updates.
    let before_hot = Signal::new(false);
    let on_hot = Signal::new(false);
    let after_hot = Signal::new(false);

    rsx! {
        div {
            // ── Insert-before drop strip ─────────────────────────────
            // Only meaningful if this node has a parent (i.e., is not
            // the root, which can't have siblings). Absolute-positioned
            // strips would be cleaner but also require the container
            // to be position:relative, so use a simple inline strip
            // whose height is 0 when no drag is in flight.
            if parent_id.is_some() {
                {drop_strip(
                    __scope, drag, before_hot, drag_active, indent, DropKind::Before,
                    DropTarget { node_id, parent_id, sibling_index, children_count: 0 },
                    cmd_tx,
                )}
            }

            // ── Row body (wrapped in a ContextMenu) ──────────────────
            ContextMenu {
                ContextMenuTarget {
                    div {
                        // draggable on anything that has a parent. Root
                        // isn't draggable (move_to rejects it engine-side,
                        // but there's also no meaningful drop target).
                        draggable: {move || if parent_id.is_some() { "true" } else { "false" }},
                        ondragstart: move || {
                            if parent_id.is_some() {
                                drag.set(node_id);
                            }
                        },
                        ondragend: move || {
                            drag.clear();
                        },
                        // Row-body acts as a drop-on target = reparent
                        // as last child (only meaningful for combinators).
                        // Self-drop is a no-op: engine-side move_to
                        // returns false for "is root" / "would cycle",
                        // but self-on-self isn't covered so filter here.
                        ondragenter: move || {
                            if drag.is_active()
                                && !is_leaf.get()
                                && drag.get() != Some(node_id)
                            {
                                on_hot.set(true);
                            }
                        },
                        ondragleave: move || {
                            on_hot.set(false);
                        },
                        ondrop: move || {
                            on_hot.set(false);
                            if is_leaf.get() { return; }
                            if let Some(src) = drag.take() {
                                if src == node_id { return; }
                                let child_count = children.get().len() as u32;
                                let _ = cmd_tx.get().send(rkp_engine::EngineCommand::MoveProceduralNode {
                                    node_id: src,
                                    new_parent_id: node_id,
                                    index: child_count,
                                });
                            }
                        },
                        style: {move || {
                            let sel = selected_node.get() == Some(node_id);
                            let hot = on_hot.get();
                            let bg = if hot {
                                "background:#1a2a3a;outline:1px dashed #4fc3f7;"
                            } else if sel {
                                "background:#37373d;color:#fff;"
                            } else {
                                "color:#ccc;"
                            };
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
                        if !is_leaf.get() || is_root.get() {
                            {render_add_child_menu(__scope, node_id, node_kind.get(), cmd_tx)}
                        }
                    }
                }
                ContextMenuDropdown {
                    {context_menu_item(__scope, "Duplicate", TablerIcon::Copy, {
                        let cmd_tx = cmd_tx;
                        move || {
                            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::DuplicateProceduralNode {
                                node_id,
                            });
                        }
                    })}
                    // Delete hidden on root — arena rejects, but hide to
                    // avoid offering the user an action that silently
                    // does nothing.
                    if !is_root.get() {
                        {context_menu_item(__scope, "Delete", TablerIcon::Trash, {
                            let cmd_tx = cmd_tx;
                            move || {
                                let _ = cmd_tx.get().send(rkp_engine::EngineCommand::RemoveProceduralNode {
                                    node_id,
                                });
                            }
                        })}
                    }
                    // "Change to …" — only on combinators. Shows the
                    // two kinds this node isn't, so the menu is the
                    // shortest one-click path to flipping Union↔Intersect
                    // etc. Especially useful on the root: the only way
                    // to make an Intersect the top-level op once the
                    // auto-promote has wrapped everything in a Union.
                    if matches!(
                        node_kind.get(),
                        ProceduralNodeKind::Union
                            | ProceduralNodeKind::Intersect
                            | ProceduralNodeKind::Subtract
                    ) {
                        DropdownMenuDivider {}
                        if !matches!(node_kind.get(), ProceduralNodeKind::Union) {
                            {context_menu_item(__scope, "Change to Union", TablerIcon::CirclePlus, {
                                let cmd_tx = cmd_tx;
                                move || {
                                    let _ = cmd_tx.get().send(
                                        rkp_engine::EngineCommand::SetProceduralNodeCombinator {
                                            node_id,
                                            kind: "Union".to_string(),
                                        },
                                    );
                                }
                            })}
                        }
                        if !matches!(node_kind.get(), ProceduralNodeKind::Intersect) {
                            {context_menu_item(__scope, "Change to Intersect", TablerIcon::CircleDot, {
                                let cmd_tx = cmd_tx;
                                move || {
                                    let _ = cmd_tx.get().send(
                                        rkp_engine::EngineCommand::SetProceduralNodeCombinator {
                                            node_id,
                                            kind: "Intersect".to_string(),
                                        },
                                    );
                                }
                            })}
                        }
                        if !matches!(node_kind.get(), ProceduralNodeKind::Subtract) {
                            {context_menu_item(__scope, "Change to Subtract", TablerIcon::CircleMinus, {
                                let cmd_tx = cmd_tx;
                                move || {
                                    let _ = cmd_tx.get().send(
                                        rkp_engine::EngineCommand::SetProceduralNodeCombinator {
                                            node_id,
                                            kind: "Subtract".to_string(),
                                        },
                                    );
                                }
                            })}
                        }
                    }
                }
            }

            // ── Children ─────────────────────────────────────────────
            if !collapsed.get() {
                for (idx, child_id) in children.get().into_iter().enumerate() {
                    {render_tree_node(
                        __scope, snapshot, child_id, depth + 1,
                        Some(node_id), idx as u32,
                        selected_node, cmd_tx, drag,
                    )}
                }
            }

            // ── Insert-after drop strip ──────────────────────────────
            // Placed AFTER the children block so it visually sits at
            // the bottom edge of the entire subtree — the drop there
            // means "insert as next sibling of this node, below its
            // subtree," not "insert as last child."
            if parent_id.is_some() {
                {drop_strip(
                    __scope, drag, after_hot, drag_active, indent, DropKind::After,
                    DropTarget { node_id, parent_id, sibling_index, children_count: 0 },
                    cmd_tx,
                )}
            }
        }
    }
}

#[derive(Copy, Clone)]
enum DropKind {
    Before,
    After,
}

#[derive(Copy, Clone)]
struct DropTarget {
    node_id: u32,
    parent_id: Option<u32>,
    sibling_index: u32,
    children_count: u32,
}

/// Render a thin sibling-insertion drop strip. 0-height when no drag
/// is active (so it doesn't visually leak into normal tree rendering),
/// 6px and lit up when the drag enters it.
#[allow(clippy::too_many_arguments)]
fn drop_strip(
    __scope: &mut Scope,
    drag: DragContext<u32>,
    hot: Signal<bool>,
    drag_active: Memo<bool>,
    indent: f32,
    kind: DropKind,
    target: DropTarget,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> Node {
    rsx! {
        div {
            style: {move || {
                if !drag_active.get() {
                    // No drag in flight → strip has no footprint.
                    "height:0;".to_string()
                } else if hot.get() {
                    format!(
                        "height:6px;margin:1px 0;margin-left:{:.0}px;\
                         background:#4fc3f7;border-radius:2px;\
                         transition:background 0.05s;",
                        8.0 + indent,
                    )
                } else {
                    format!(
                        "height:6px;margin:1px 0;margin-left:{:.0}px;\
                         background:transparent;",
                        8.0 + indent,
                    )
                }
            }},
            ondragenter: move || {
                if !drag.is_active() { return; }
                if drag.get() == Some(target.node_id) { return; }
                hot.set(true);
            },
            ondragleave: move || {
                hot.set(false);
            },
            ondrop: move || {
                hot.set(false);
                // No-op if dragging onto self (arena would reject anyway).
                let src = match drag.take() {
                    Some(s) if s != target.node_id => s,
                    _ => return,
                };
                let Some(new_parent) = target.parent_id else { return; };
                let idx = match kind {
                    DropKind::Before => target.sibling_index,
                    DropKind::After => target.sibling_index + 1,
                };
                let _ = cmd_tx.get().send(rkp_engine::EngineCommand::MoveProceduralNode {
                    node_id: src,
                    new_parent_id: new_parent,
                    index: idx,
                });
                let _ = target.children_count;
            },
        }
    }
}

/// A single context-menu row. Thin wrapper around `DropdownMenuItem`
/// so icon + label styling come from rinch's theme and the parent
/// ContextMenu's close-on-click behavior works automatically (it sets
/// a thread-local close signal that DropdownMenuItem flips on click).
fn context_menu_item(
    __scope: &mut Scope,
    label: &'static str,
    icon: TablerIcon,
    on_click: impl Fn() + 'static + Clone,
) -> Node {
    rsx! {
        DropdownMenuItem {
            left_section: icon,
            onclick: move || on_click(),
            {label}
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
                // bottom_start anchors the dropdown's left edge to the +
                // button's left edge so the menu grows rightward — the
                // tree now lives floating on the left side of the build
                // viewport, so there's more room to the right.
                position: "bottom_start",
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
    // Popover — unlike ContextMenu — does NOT install the thread-local
    // close signal, so DropdownMenuItem can't auto-close it. We flip
    // `opened` ourselves inside the onclick after dispatching the
    // command.
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
