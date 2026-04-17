//! Viewport component — renders the engine's output and forwards input events.

use rinch::prelude::*;
use rinch::render_surface::{RenderSurface, SurfaceEvent, SurfaceMouseButton};

use crate::CommandSender;
use crate::ui::store::EditorStore;
use super::viewport_toolbar::{ViewportHeaderBar, EditModeToolbar};

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
                width: w, height: h,
            });
        }

        match event {
            MouseMove { x, y } => {
                let dx = x - last_mx.get();
                let dy = y - last_my.get();
                last_mx.set(x);
                last_my.set(y);
                let _ = cmd_tx.send(rkp_engine::EngineCommand::MouseMove { x, y, dx, dy });
            }
            MouseDown { button, x, y } => {
                let _ = cmd_tx.send(rkp_engine::EngineCommand::MouseButton {
                    button: map_button(button),
                    pressed: true,
                });
                // Left click → pick object at this pixel.
                if button == SurfaceMouseButton::Left {
                    let _ = cmd_tx.send(rkp_engine::EngineCommand::Pick {
                        x: x as u32,
                        y: y as u32,
                    });
                }
            }
            MouseUp { button, .. } => {
                let _ = cmd_tx.send(rkp_engine::EngineCommand::MouseButton {
                    button: map_button(button),
                    pressed: false,
                });
            }
            MouseWheel { delta_y, .. } => {
                let _ = cmd_tx.send(rkp_engine::EngineCommand::Scroll { delta: delta_y });
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
                if let Some(key) = map_key(&key_data.code) {
                    let _ = cmd_tx.send(rkp_engine::EngineCommand::KeyDown { key });
                }
            }
            KeyUp(key_data) => {
                if let Some(key) = map_key(&key_data.code) {
                    let _ = cmd_tx.send(rkp_engine::EngineCommand::KeyUp { key });
                }
            }
            Drop { x, y } => {
                // Model drag-and-drop: place model at drop position.
                if let Some(model_path) = store.model_drag.get() {
                    let _ = cmd_tx.send(rkp_engine::EngineCommand::LoadAsset {
                        path: model_path,
                        position: glam::Vec3::ZERO, // TODO: raycast to ground plane
                    });
                    store.model_drag.set(None);
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
            DragEnter { .. } | DragOver { .. } | DragLeave => {
                // Accept model drags silently.
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
            }
        }
    }
}
