//! Viewport toolbars — fixed header bar + floating gizmo overlay.
//!
//! The fixed bar holds viewport-wide controls (play/stop, camera selectors, etc.)
//! that are always visible. The floating overlay holds mode-specific tools
//! (translate/rotate/scale) that only apply during editing.

use rinch::prelude::*;
use rinch_tabler_icons::{TablerIcon, TablerIconStyle, render_tabler_icon};

use rkp_engine::gizmo::GizmoMode;

use crate::CommandSender;
use crate::ui::store::EditorStore;

// ── Fixed header bar ────────────────────────────────────────────────────

/// Fixed toolbar row above the render surface — viewport-wide controls.
#[component]
pub fn ViewportHeaderBar() -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd = use_context::<CommandSender>();

    rsx! {
        div {
            style: "height:32px;display:flex;align-items:center;padding:0 6px;\
                    background:#252526;border-bottom:1px solid #3c3c3c;flex-shrink:0;",

            // Play/Stop (left-aligned)
            {play_button(__scope, store, cmd)}
        }
    }
}

fn play_button(
    __scope: &mut rinch::core::dom::RenderScope,
    store: EditorStore,
    cmd: CommandSender,
) -> rinch::core::dom::NodeHandle {
    rsx! {
        div {
            style: {move || {
                if store.play_mode.get() {
                    "display:flex;align-items:center;gap:4px;padding:2px 10px;\
                     background:#5a2d2d;border:1px solid #8b3a3a;border-radius:4px;\
                     cursor:pointer;color:#ef5350;font-size:11px;font-weight:600;"
                } else {
                    "display:flex;align-items:center;gap:4px;padding:2px 10px;\
                     background:#2d5a2d;border:1px solid #3c7c3c;border-radius:4px;\
                     cursor:pointer;color:#4caf50;font-size:11px;font-weight:600;"
                }
            }},
            title: {move || if store.play_mode.get() { "Stop (F5)" } else { "Play (F5)" }},
            onclick: {
                let cmd = cmd.clone();
                move || {
                    if store.play_mode.get() {
                        let _ = cmd.0.send(rkp_engine::EngineCommand::PlayStop);
                    } else {
                        let _ = cmd.0.send(rkp_engine::EngineCommand::PlayStart);
                    }
                }
            },
            span {
                style: "width:14px;height:14px;display:inline-flex;\
                        align-items:center;justify-content:center;",
                if store.play_mode.get() {
                    {render_tabler_icon(__scope, TablerIcon::PlayerStop, TablerIconStyle::Filled)}
                }
                if !store.play_mode.get() {
                    {render_tabler_icon(__scope, TablerIcon::PlayerPlay, TablerIconStyle::Filled)}
                }
            }
            if store.play_mode.get() {
                {"Stop"}
            }
            if !store.play_mode.get() {
                {"Play"}
            }
        }
    }
}

// ── Floating gizmo overlay ──────────────────────────────────────────────

/// Floating pill overlay for mode-specific editing tools (translate/rotate/scale).
#[component]
pub fn EditModeToolbar() -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd = use_context::<CommandSender>();

    rsx! {
        div {
            style: "position:absolute;top:8px;left:8px;z-index:20;\
                    display:flex;flex-direction:column;gap:1px;\
                    background:rgba(30,30,30,0.85);border-radius:6px;\
                    border:1px solid #3c3c3c;padding:2px;backdrop-filter:blur(8px);",

                {gizmo_button(
                    __scope,
                    TablerIcon::ArrowsMove,
                    "Translate (W)",
                    Memo::new(move || store.gizmo_mode.get() == GizmoMode::Translate),
                    {
                        let cmd = cmd.clone();
                        move || {
                            store.gizmo_mode.set(GizmoMode::Translate);
                            let _ = cmd.0.send(rkp_engine::EngineCommand::SetGizmoMode {
                                mode: GizmoMode::Translate,
                            });
                        }
                    },
                )}
                {gizmo_button(
                    __scope,
                    TablerIcon::Rotate,
                    "Rotate (E)",
                    Memo::new(move || store.gizmo_mode.get() == GizmoMode::Rotate),
                    {
                        let cmd = cmd.clone();
                        move || {
                            store.gizmo_mode.set(GizmoMode::Rotate);
                            let _ = cmd.0.send(rkp_engine::EngineCommand::SetGizmoMode {
                                mode: GizmoMode::Rotate,
                            });
                        }
                    },
                )}
                {gizmo_button(
                    __scope,
                    TablerIcon::Resize,
                    "Scale (R)",
                    Memo::new(move || store.gizmo_mode.get() == GizmoMode::Scale),
                    {
                        let cmd = cmd.clone();
                        move || {
                            store.gizmo_mode.set(GizmoMode::Scale);
                            let _ = cmd.0.send(rkp_engine::EngineCommand::SetGizmoMode {
                                mode: GizmoMode::Scale,
                            });
                        }
                    },
                )}
        }
    }
}

fn gizmo_button(
    __scope: &mut rinch::core::dom::RenderScope,
    icon: TablerIcon,
    tooltip: &str,
    is_active: Memo<bool>,
    on_click: impl Fn() + Clone + 'static,
) -> rinch::core::dom::NodeHandle {
    let tooltip = tooltip.to_string();
    rsx! {
        div {
            style: {move || {
                if is_active.get() {
                    "width:28px;height:28px;display:flex;align-items:center;\
                     justify-content:center;border-radius:4px;cursor:pointer;\
                     background:#4fc3f7;color:#1e1e1e;"
                } else {
                    "width:28px;height:28px;display:flex;align-items:center;\
                     justify-content:center;border-radius:4px;cursor:pointer;\
                     color:#999;background:transparent;"
                }
            }},
            title: tooltip,
            onclick: move || on_click(),
            span {
                style: "width:18px;height:18px;display:inline-flex;\
                        align-items:center;justify-content:center;",
                {render_tabler_icon(__scope, icon, TablerIconStyle::Outline)}
            }
        }
    }
}
