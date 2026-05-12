//! Viewport component — renders the engine's output and forwards input events.

use rinch::prelude::*;
use rinch::render_surface::{RenderSurface, SurfaceEvent, SurfaceMouseButton};

use rkp_engine::viewport::ViewportId;

use crate::CommandSender;
use crate::ui::store::EditorStore;
use super::viewport_toolbar::{ViewportHeaderBar, EditModeToolbar, BrushToolbar};

/// The viewport id this panel renders. Phase 3: only the MAIN viewport has
/// a UI panel; the build viewport gets its own component in Phase 6.
const PANEL_VIEWPORT: ViewportId = ViewportId::MAIN;

/// Map rinch SurfaceMouseButton to rkp_runtime InputMouseButton.
fn map_button(btn: SurfaceMouseButton) -> rkp_runtime::input::InputMouseButton {
    match btn {
        SurfaceMouseButton::Left => rkp_runtime::input::InputMouseButton::Left,
        SurfaceMouseButton::Right => rkp_runtime::input::InputMouseButton::Right,
        SurfaceMouseButton::Middle => rkp_runtime::input::InputMouseButton::Middle,
    }
}

/// Map a rinch key code string to InputKeyCode.
fn map_key(code: &str) -> Option<rkp_runtime::input::InputKeyCode> {
    use rkp_runtime::input::InputKeyCode::*;
    Some(match code {
        "KeyA" => A, "KeyB" => B, "KeyC" => C, "KeyD" => D,
        "KeyE" => E, "KeyF" => F, "KeyG" => G, "KeyH" => H,
        "KeyI" => I, "KeyJ" => J, "KeyK" => K, "KeyL" => L,
        "KeyM" => M, "KeyN" => N, "KeyO" => O, "KeyP" => P,
        "KeyQ" => Q, "KeyR" => R, "KeyS" => S, "KeyT" => T,
        "KeyU" => U, "KeyV" => V, "KeyW" => W, "KeyX" => X,
        "KeyY" => Y, "KeyZ" => Z,
        "Digit0" => Num0, "Digit1" => Num1, "Digit2" => Num2,
        "Digit3" => Num3, "Digit4" => Num4, "Digit5" => Num5,
        "Digit6" => Num6, "Digit7" => Num7, "Digit8" => Num8,
        "Digit9" => Num9,
        "Space" => Space,
        "ShiftLeft" => ShiftLeft, "ShiftRight" => ShiftRight,
        "ControlLeft" => ControlLeft, "ControlRight" => ControlRight,
        "AltLeft" => AltLeft, "AltRight" => AltRight,
        "Tab" => Tab, "Escape" => Escape, "Enter" => Enter,
        "Backspace" => Backspace, "Delete" => Delete,
        "ArrowUp" => ArrowUp, "ArrowDown" => ArrowDown,
        "ArrowLeft" => ArrowLeft, "ArrowRight" => ArrowRight,
        "F1" => F1, "F2" => F2, "F3" => F3, "F4" => F4,
        "F5" => F5, "F6" => F6, "F7" => F7, "F8" => F8,
        "F9" => F9, "F10" => F10, "F11" => F11, "F12" => F12,
        _ => return None,
    })
}

#[component]
pub fn Viewport() -> NodeHandle {
    let surface = use_context::<RenderSurfaceHandle>();
    let cmd = use_context::<CommandSender>();
    let store = use_context::<EditorStore>();

    // Track last mouse position for computing deltas.
    let last_mx = std::cell::Cell::new(0.0f32);
    let last_my = std::cell::Cell::new(0.0f32);
    // LMB-held state — paint mode uses this to decide whether a
    // MouseMove event should fire another `PaintAtPixel` stamp.
    let lmb_held = std::cell::Cell::new(false);

    let cmd_tx = cmd.0.clone();
    let surface_for_handler = surface.clone();
    surface.set_event_handler(move |event| {
        use SurfaceEvent::*;

        // Check if surface size changed — send Resize to engine.
        {
            let (w, h) = surface_for_handler.layout_size();
            let w = w.max(64);
            let h = h.max(64);
            // Only send resize occasionally — every mouse event is fine,
            // the engine no-ops if the size hasn't changed.
            let _ = cmd_tx.send(rkp_engine::EngineCommand::Resize {
                id: PANEL_VIEWPORT, width: w, height: h,
            });
        }

        // Small helper that reads the current paint signals out of the
        // store and sends a `PaintAtPixel` for the given pixel. The
        // engine coalesces picks (only one in-flight at a time) so
        // firing this on every MouseMove during a drag is safe —
        // intermediate requests get replaced, not queued.
        let send_paint_stamp = |x: f32, y: f32| {
            let _ = cmd_tx.send(rkp_engine::EngineCommand::PaintAtPixel {
                id: PANEL_VIEWPORT,
                x: x.max(0.0) as u32,
                y: y.max(0.0) as u32,
                radius: store.paint_radius.get(),
                color: store.paint_color.get(),
                strength: store.paint_strength.get(),
                falloff: store.paint_falloff.get(),
                mode: store.paint_mode.get(),
                material_id: store.selected_material.get().unwrap_or(0),
            });
        };

        // Sculpt equivalent — same single-in-flight coalescing as paint.
        let send_sculpt_stamp = |x: f32, y: f32| {
            let _ = cmd_tx.send(rkp_engine::EngineCommand::SculptAtPixel {
                id: PANEL_VIEWPORT,
                x: x.max(0.0) as u32,
                y: y.max(0.0) as u32,
                radius: store.sculpt_radius.get(),
                falloff: store.sculpt_falloff.get(),
                mode: store.sculpt_mode.get(),
                material_id: store.selected_material.get().unwrap_or(0),
            });
        };

        match event {
            MouseMove { x, y } => {
                let dx = x - last_mx.get();
                let dy = y - last_my.get();
                last_mx.set(x);
                last_my.set(y);
                let _ = cmd_tx.send(rkp_engine::EngineCommand::MouseMove {
                    id: PANEL_VIEWPORT, x, y, dx, dy,
                });
                // Paint-drag: in paint mode with LMB held, each mouse
                // move issues another stamp. The engine's pick
                // in-flight gate naturally coalesces duplicates.
                if store.paint_active.get() && lmb_held.get() {
                    // Drag-paint: each mouse move issues another stamp
                    // while LMB is held. Hover tracking is GPU-driven
                    // (the brush-state probe pass reads gbuf at the
                    // engine-side `mouse_pos`), so no extra command.
                    send_paint_stamp(x, y);
                } else if store.sculpt_active.get() && lmb_held.get() {
                    send_sculpt_stamp(x, y);
                }
            }
            MouseDown { button, x, y } => {
                let _ = cmd_tx.send(rkp_engine::EngineCommand::MouseButton {
                    id: PANEL_VIEWPORT,
                    button: map_button(button),
                    pressed: true,
                });
                if button == SurfaceMouseButton::Left {
                    lmb_held.set(true);
                    if store.paint_active.get() {
                        // Paint mode owns LMB — suppress the normal
                        // click-select pick so the click doesn't
                        // deselect everything while painting.
                        send_paint_stamp(x, y);
                    } else if store.sculpt_active.get() {
                        // Sculpt owns LMB the same way paint does.
                        send_sculpt_stamp(x, y);
                    } else {
                        let _ = cmd_tx.send(rkp_engine::EngineCommand::Pick {
                            id: PANEL_VIEWPORT,
                            x: x as u32,
                            y: y as u32,
                        });
                    }
                }
            }
            MouseUp { button, .. } => {
                let _ = cmd_tx.send(rkp_engine::EngineCommand::MouseButton {
                    id: PANEL_VIEWPORT,
                    button: map_button(button),
                    pressed: false,
                });
                if button == SurfaceMouseButton::Left {
                    lmb_held.set(false);
                }
            }
            MouseWheel { delta_y, .. } => {
                let _ = cmd_tx.send(rkp_engine::EngineCommand::Scroll {
                    id: PANEL_VIEWPORT, delta: delta_y,
                });
            }
            KeyDown(key_data) => {
                // Delete key → delete selected entity.
                if key_data.code == "Delete" || key_data.code == "Backspace" {
                    let _ = cmd_tx.send(rkp_engine::EngineCommand::DeleteSelected);
                }
                // F5 → toggle play mode.
                if key_data.code == "F5" {
                    if store.play_mode.get() {
                        let _ = cmd_tx.send(rkp_engine::EngineCommand::PlayStop);
                    } else {
                        let _ = cmd_tx.send(rkp_engine::EngineCommand::PlayStart);
                    }
                }
                // P → toggle paint mode. Same signal the PaintToolbar
                // button writes, so the overlay updates immediately,
                // and we mirror the SetPaintActive command so the
                // engine lights up the cursor wireframe.
                if key_data.code == "KeyP" {
                    let on = !store.paint_active.get();
                    store.paint_active.set(on);
                    let _ = cmd_tx.send(rkp_engine::EngineCommand::SetPaintActive {
                        active: on,
                        radius: store.paint_radius.get(),
                    });
                }
                if let Some(key) = map_key(&key_data.code) {
                    let _ = cmd_tx.send(rkp_engine::EngineCommand::KeyDown { key });
                }
            }
            KeyUp(key_data) => {
                if let Some(key) = map_key(&key_data.code) {
                    let _ = cmd_tx.send(rkp_engine::EngineCommand::KeyUp { key });
                }
            }
            Drop { .. } => {
                // Any of the three drag sources (model / generator /
                // preset) spawned a live preview on DragEnter and kept
                // it chasing the cursor. Commit retires the preview so
                // the entity stops tracking further picks and stays
                // where it last landed.
                let had_preview_drag = store.model_drag.get().is_some()
                    || store.generator_drag.get().is_some()
                    || store.generator_preset_drag.get().is_some();
                if had_preview_drag {
                    let _ = cmd_tx.send(rkp_engine::EngineCommand::DragPreviewCommit);
                    store.model_drag.set(None);
                    store.generator_drag.set(None);
                    store.generator_preset_drag.set(None);
                }
                // Material drag-and-drop: assign to selected entity.
                if let Some(mat_id) = store.material_drag.get() {
                    if let Some(entity_id) = store.selected_entity.get() {
                        let _ = cmd_tx.send(rkp_engine::EngineCommand::AssignMaterial {
                            entity_id,
                            material_id: mat_id,
                        });
                    }
                    store.material_drag.set(None);
                }
            }
            DragEnter { x, y } => {
                let px = (x as i32).max(0) as u32;
                let py = (y as i32).max(0) as u32;
                // Pick the first drag source that's active. Only one
                // of these can be set at a time (ondragstart sets one).
                let source = if let Some(path) = store.model_drag.get() {
                    Some(rkp_engine::DragPreviewSource::Asset { path })
                } else if let Some(name) = store.generator_drag.get() {
                    Some(rkp_engine::DragPreviewSource::Generator { name })
                } else if let Some(path) = store.generator_preset_drag.get() {
                    Some(rkp_engine::DragPreviewSource::GeneratorPreset { path })
                } else {
                    None
                };
                if let Some(source) = source {
                    // Hide rinch's built-in HTML drag ghost — the 3D
                    // preview in the viewport is the authoritative
                    // visual while the cursor is over this surface.
                    // Restored on DragLeave (which fires both when the
                    // cursor exits and after a successful Drop).
                    rinch::prelude::suppress_drag_ghost();
                    let _ = cmd_tx.send(rkp_engine::EngineCommand::DragPreviewEnter {
                        id: PANEL_VIEWPORT,
                        source,
                        x: px, y: py,
                    });
                }
            }
            DragOver { x, y } => {
                let any_active = store.model_drag.get().is_some()
                    || store.generator_drag.get().is_some()
                    || store.generator_preset_drag.get().is_some();
                if any_active {
                    let _ = cmd_tx.send(rkp_engine::EngineCommand::DragPreviewOver {
                        id: PANEL_VIEWPORT,
                        x: (x as i32).max(0) as u32,
                        y: (y as i32).max(0) as u32,
                    });
                }
            }
            DragLeave => {
                // `DragLeave` also fires after a successful `Drop`. The
                // Drop handler clears every drag signal, so on a valid
                // drop none of these checks pass and cancel is skipped.
                // A true drag-out-without-drop still has the signal set
                // and aborts the preview.
                let any_active = store.model_drag.get().is_some()
                    || store.generator_drag.get().is_some()
                    || store.generator_preset_drag.get().is_some();
                if any_active {
                    let _ = cmd_tx.send(rkp_engine::EngineCommand::DragPreviewCancel);
                }
                // Restore the HTML ghost either way (see DragEnter).
                rinch::prelude::restore_drag_ghost();
            }
            _ => {}
        }
    });

    rsx! {
        div {
            style: "display:flex;flex-direction:column;width:100%;height:100%;",
            // Fixed header — viewport-wide controls (play/stop, etc.)
            ViewportHeaderBar {}
            // Render area with floating gizmo overlay
            div {
                style: "flex:1;min-height:0;position:relative;",
                RenderSurface { surface: Some(surface.clone()) }
                EditModeToolbar {}
                BrushToolbar {}
            }
        }
    }
}
