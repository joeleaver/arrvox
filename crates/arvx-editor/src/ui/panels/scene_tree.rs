//! Scene tree panel — hierarchical view of scene objects.
//!
//! Renders root objects (no parent) at the top level, with children
//! nested below their parents. Right-click for context menu:
//! Duplicate, Delete, and — for procedural objects — "Copy to New
//! Voxel Object" and "Convert to Voxel Object" (the latter stages
//! the entity into `store.convert_procedural_target`; the
//! confirmation modal is mounted by `LayoutRoot` so the dialog can
//! catch clicks even when the scene-tree panel is narrow).

use rinch::prelude::*;
use rinch_tabler_icons::{TablerIcon, TablerIconStyle, render_tabler_icon};

use crate::CommandSender;
use crate::ui::store::{DropZone, EditorStore};
use arvx_engine::SceneObjectInfo;

/// Resolve `(new_parent, new_order)` from a drop. Reads the latest
/// `store.objects` snapshot (already sorted by `tree_order`) and
/// interpolates between the target's neighbors at the resolved parent
/// level.
///
/// Drop zones map as follows:
///   * `Before` → drop above target, at target's level.
///   * `After` with no expanded children → below target, at its level.
///   * `After` with expanded children → as target's first child.
///   * `Inside` → append as target's last child.
///
/// `target == None` + zone=Inside means "scene root" (from the Scene
/// row). Before/After on root are treated the same as Inside for
/// single-scene setups — editor shouldn't send them currently.
fn compute_drop(
    objects: &[SceneObjectInfo],
    target: Option<uuid::Uuid>,
    zone: DropZone,
    target_has_expanded_children: bool,
) -> Option<(Option<uuid::Uuid>, f64)> {
    // Helper: given a parent filter + an order bound direction,
    // find the order of the closest neighbor at the same level.
    let max_child_order = |parent: Option<uuid::Uuid>| -> Option<f64> {
        objects
            .iter()
            .filter(|o| o.parent_id == parent)
            .map(|o| o.tree_order)
            .fold(None, |acc, o| Some(acc.map_or(o, |a: f64| a.max(o))))
    };
    let min_child_order = |parent: Option<uuid::Uuid>| -> Option<f64> {
        objects
            .iter()
            .filter(|o| o.parent_id == parent)
            .map(|o| o.tree_order)
            .fold(None, |acc, o| Some(acc.map_or(o, |a: f64| a.min(o))))
    };

    // Scene-root drop (no target entity) → append to end of roots.
    let Some(target_id) = target else {
        let order = max_child_order(None).map(|m| m + 1.0).unwrap_or(0.0);
        return Some((None, order));
    };

    let target_obj = objects.iter().find(|o| o.id == target_id)?;
    let target_parent = target_obj.parent_id;
    let target_order = target_obj.tree_order;

    match zone {
        DropZone::Before => {
            // Previous sibling at target's level: largest order below
            // target's in target_parent's children.
            let prev = objects
                .iter()
                .filter(|o| o.parent_id == target_parent && o.tree_order < target_order)
                .map(|o| o.tree_order)
                .fold(None, |acc, o| Some(acc.map_or(o, |a: f64| a.max(o))));
            let order = prev
                .map(|p| (p + target_order) * 0.5)
                .unwrap_or(target_order - 1.0);
            Some((target_parent, order))
        }
        DropZone::After if target_has_expanded_children => {
            // "After expanded parent" = first child of target. Place
            // below target's existing children's minimum so it sorts
            // first among siblings under target.
            let first_child_order = min_child_order(Some(target_id));
            let order = first_child_order
                .map(|m| m - 1.0)
                .unwrap_or(target_order + 1.0);
            Some((Some(target_id), order))
        }
        DropZone::After => {
            // Next sibling at target's level.
            let next = objects
                .iter()
                .filter(|o| o.parent_id == target_parent && o.tree_order > target_order)
                .map(|o| o.tree_order)
                .fold(None, |acc, o| Some(acc.map_or(o, |a: f64| a.min(o))));
            let order = next
                .map(|n| (target_order + n) * 0.5)
                .unwrap_or(target_order + 1.0);
            Some((target_parent, order))
        }
        DropZone::Inside => {
            // Append as target's last child.
            let order = max_child_order(Some(target_id))
                .map(|m| m + 1.0)
                .unwrap_or(target_order + 1.0);
            Some((Some(target_id), order))
        }
    }
}

#[component]
pub fn SceneTree() -> NodeHandle {
    let store = use_context::<EditorStore>();

    let roots = Memo::new(move || {
        store.objects.get().into_iter()
            .filter(|o| o.parent_id.is_none())
            .collect::<Vec<_>>()
    });

    let cmd_tx = Signal::new(use_context::<CommandSender>().0);
    rsx! {
        div {
            style: "display:flex;flex-direction:column;height:100%;overflow-y:auto;",
            div {
                style: "flex:1;padding:4px;",
                // Synthetic scene-root row. Not an engine entity —
                // purely a drop target that reparents to the scene
                // root (clears Parent) when an entity is dragged
                // Inside it. Before/After on this row are ignored
                // because there's currently only one scene; future
                // multi-scene support can make this per-scene and
                // re-enable sibling insertion among scene roots.
                {scene_root_row(__scope, store, cmd_tx)}
                for obj in roots.get() {
                    {tree_node(__scope, &obj, 1, store)}
                }
            }
        }
    }
}

/// The "Scene" drop-only row rendered at the top of the tree. Accepts
/// `Inside` drops to reparent an entity to the root (clear its
/// `Parent` component) and append at the end.
fn scene_root_row(
    __scope: &mut rinch::core::dom::RenderScope,
    store: EditorStore,
    cmd_tx: Signal<crossbeam::channel::Sender<arvx_engine::EngineCommand>>,
) -> rinch::core::dom::NodeHandle {
    rsx! {
        div {
            style: {
                move || {
                    // Full-row tint when the current drop hint points
                    // at "root" (we encode that as `None` in the hint).
                    // The inner Option value of the tuple is
                    // `Option<(Uuid, DropPosition)>`; to avoid teaching
                    // the hint a third state we simply match on "no
                    // hint set AND dragging" as the indicator here.
                    let show_target = store.scene_tree_root_hint.get();
                    let hint_bg = if show_target {
                        "background:rgba(79,195,247,0.25);color:#fff;"
                    } else {
                        "color:#aaa;"
                    };
                    format!(
                        "display:flex;align-items:center;padding:2px 8px;\
                         border-radius:3px;font-size:12px;gap:4px;\
                         font-weight:600;text-transform:uppercase;letter-spacing:0.05em;\
                         {hint_bg}"
                    )
                }
            },
            ondragover: move || {
                if store.scene_tree_drag.get().is_some() {
                    if !store.scene_tree_root_hint.get() {
                        store.scene_tree_root_hint.set(true);
                    }
                }
            },
            ondragleave: move || {
                if store.scene_tree_root_hint.get() {
                    store.scene_tree_root_hint.set(false);
                }
            },
            ondrop: move || {
                let Some(source_id) = store.scene_tree_drag.get() else { return };
                store.scene_tree_root_hint.set(false);
                let objects = store.objects.get();
                // Scene-root drop: `target = None`, `Inside` zone —
                // `compute_drop` short-circuits to "append to end of
                // roots" (max root-order + 1.0).
                if let Some((new_parent, new_order)) = compute_drop(
                    &objects, None, DropZone::Inside, false,
                ) {
                    let _ = cmd_tx.get().send(
                        arvx_engine::EngineCommand::ReorderEntity {
                            entity: source_id,
                            new_parent,
                            new_order,
                        },
                    );
                }
                store.scene_tree_drag.set(None);
            },
            span { style: "width:12px;" }
            span {
                style: "width:14px;height:14px;display:inline-flex;align-items:center;\
                        justify-content:center;flex-shrink:0;color:#999;",
                {render_tabler_icon(__scope, TablerIcon::Hierarchy2, TablerIconStyle::Outline)}
            }
            span { "Scene" }
        }
    }
}

/// Render a single tree node + its children recursively.
fn tree_node(
    __scope: &mut rinch::core::dom::RenderScope,
    object: &SceneObjectInfo,
    depth: u32,
    store: EditorStore,
) -> rinch::core::dom::NodeHandle {
    let cmd_tx = Signal::new(use_context::<CommandSender>().0);
    let id = object.id;
    let name = object.name.clone();
    let is_procedural = object.is_procedural;
    let collapsed = Signal::new(false);
    let indent = depth as f32 * 16.0;

    let icon = if object.is_camera { TablerIcon::Camera }
        else if object.is_light { TablerIcon::Bulb }
        else { TablerIcon::Cube };

    let children = Memo::new(move || {
        store.objects.get().into_iter()
            .filter(|o| o.parent_id == Some(id))
            .collect::<Vec<_>>()
    });
    let has_children = Memo::new(move || !children.get().is_empty());

    rsx! {
        div {
            // Context menu wraps the entire node
            ContextMenu {
                ContextMenuTarget {
                    // Node row
                    div {
                        style: {
                            move || {
                                let selected = store.selected_entity.get() == Some(id);
                                let sel_bg = if selected { "background:#37373d;color:#fff;" } else { "color:#ccc;" };
                                // Drop-hint visuals. A top/bottom
                                // border indicates the sibling
                                // insertion point; a full tint
                                // indicates reparent-inside. Overrides
                                // the selection background for Inside
                                // so the reparent target is clearly
                                // the cursor's focus.
                                let (border_top, border_bottom, hint_bg) = match store.scene_tree_drop_hint.get() {
                                    Some((hid, DropZone::Before)) if hid == id =>
                                        ("border-top:2px solid #4fc3f7;margin-top:-2px;", "", ""),
                                    Some((hid, DropZone::After)) if hid == id =>
                                        ("", "border-bottom:2px solid #4fc3f7;margin-bottom:-2px;", ""),
                                    Some((hid, DropZone::Inside)) if hid == id =>
                                        ("", "", "background:rgba(79,195,247,0.25);color:#fff;"),
                                    _ => ("", "", ""),
                                };
                                let bg = if hint_bg.is_empty() { sel_bg } else { hint_bg };
                                format!(
                                    "display:flex;align-items:center;padding:2px 8px 2px {:.0}px;\
                                     cursor:pointer;border-radius:3px;font-size:12px;gap:4px;\
                                     {bg}{border_top}{border_bottom}",
                                    8.0 + indent
                                )
                            }
                        },
                        onclick: move || {
                            let _ = cmd_tx.get().send(arvx_engine::EngineCommand::SelectEntity { entity_id: id });
                        },
                        draggable: "true",
                        ondragstart: move || {
                            store.scene_tree_drag.set(Some(id));
                        },
                        ondragend: move || {
                            store.scene_tree_drag.set(None);
                            store.scene_tree_drop_hint.set(None);
                        },
                        // Live drop-position feedback: compute from the
                        // cursor's fractional Y within the row and
                        // publish to the store so this row's style
                        // closure paints the indicator. Skip self-
                        // target (dragging onto self is a no-op).
                        ondragover: move || {
                            if store.scene_tree_drag.get() == Some(id) {
                                return;
                            }
                            let ctx = rinch::core::get_click_context();
                            let h = ctx.element_height.max(1.0);
                            let rel_y = (ctx.mouse_y - ctx.element_y).max(0.0) / h;
                            // 25/50/25 split — the 50 % Inside zone
                            // is ~11 px on a 22 px row, generous
                            // enough to hit without precise aim. Rows
                            // can be made even wider for Inside by
                            // gating on whether target can have
                            // children (procedurals, generators), but
                            // a fixed split is simpler.
                            let zone = if rel_y < 0.25 {
                                DropZone::Before
                            } else if rel_y > 0.75 {
                                DropZone::After
                            } else {
                                DropZone::Inside
                            };
                            if store.scene_tree_drop_hint.get() != Some((id, zone)) {
                                store.scene_tree_drop_hint.set(Some((id, zone)));
                            }
                        },
                        ondragleave: move || {
                            // Only clear if *this* row is the current
                            // hint — another row's enter may fire
                            // before our leave, so unconditional clear
                            // would race.
                            if let Some((hid, _)) = store.scene_tree_drop_hint.get() {
                                if hid == id {
                                    store.scene_tree_drop_hint.set(None);
                                }
                            }
                        },
                        ondrop: move || {
                            let Some(source_id) = store.scene_tree_drag.get() else { return };
                            // The final zone comes from the last
                            // `ondragover` hint — rinch's `ondrop`
                            // dispatch doesn't populate element bounds
                            // in the click context, so recomputing
                            // rel_y here gives garbage. Fall back to
                            // Inside if the hint got lost.
                            let zone = store.scene_tree_drop_hint.get()
                                .map(|(_, z)| z)
                                .unwrap_or(DropZone::Inside);
                            store.scene_tree_drop_hint.set(None);
                            if source_id == id { return; }
                            let has_expanded = has_children.get() && !collapsed.get();
                            let objects = store.objects.get();
                            if let Some((new_parent, new_order)) = compute_drop(
                                &objects, Some(id), zone, has_expanded,
                            ) {
                                let _ = cmd_tx.get().send(
                                    arvx_engine::EngineCommand::ReorderEntity {
                                        entity: source_id,
                                        new_parent,
                                        new_order,
                                    },
                                );
                            }
                            store.scene_tree_drag.set(None);
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
                            {render_tabler_icon(__scope, icon, TablerIconStyle::Outline)}
                        }

                        // Name
                        span { style: "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;", {name} }
                    }
                }
                ContextMenuDropdown {
                    DropdownMenuItem {
                        left_section: TablerIcon::Copy,
                        onclick: move || {
                            let _ = cmd_tx.get().send(arvx_engine::EngineCommand::SelectEntity { entity_id: id });
                            let _ = cmd_tx.get().send(arvx_engine::EngineCommand::DuplicateObject { entity_id: id });
                        },
                        "Duplicate"
                    }
                    // Procedural-only entries. Grouped together and
                    // separated from the generic actions with a
                    // divider so the destructive "Convert" item is
                    // visually bracketed and less likely to be hit
                    // by accident on the way to Delete.
                    if is_procedural {
                        DropdownMenuDivider {}
                        DropdownMenuItem {
                            left_section: TablerIcon::Copy,
                            onclick: move || {
                                let _ = cmd_tx.get().send(
                                    arvx_engine::EngineCommand::CopyProceduralToNewVoxel {
                                        entity_id: id,
                                    },
                                );
                            },
                            "Copy to New Voxel Object"
                        }
                        DropdownMenuItem {
                            left_section: TablerIcon::Cube,
                            onclick: move || {
                                // Destructive — stage the target
                                // on the store; `LayoutRoot` mounts
                                // the confirmation modal.
                                store.convert_procedural_target.set(Some(id));
                            },
                            "Convert to Voxel Object…"
                        }
                    }
                    DropdownMenuDivider {}
                    DropdownMenuItem {
                        color: "red",
                        left_section: TablerIcon::Trash,
                        onclick: move || {
                            let _ = cmd_tx.get().send(arvx_engine::EngineCommand::DeleteObject { entity_id: id });
                        },
                        "Delete"
                    }
                }
            }

            // Children
            if !collapsed.get() {
                for child in children.get() {
                    {tree_node(__scope, &child, depth + 1, store)}
                }
            }
        }
    }
}
