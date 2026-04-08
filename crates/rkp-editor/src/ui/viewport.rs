//! Viewport component — renders the engine's output and forwards input events.

use rinch::prelude::*;
use rinch::render_surface::{RenderSurface, SurfaceEvent, SurfaceMouseButton};

use crate::CommandSender;

/// Map rinch SurfaceMouseButton to rkf_runtime InputMouseButton.
fn map_button(btn: SurfaceMouseButton) -> rkf_runtime::input::InputMouseButton {
    match btn {
        SurfaceMouseButton::Left => rkf_runtime::input::InputMouseButton::Left,
        SurfaceMouseButton::Right => rkf_runtime::input::InputMouseButton::Right,
        SurfaceMouseButton::Middle => rkf_runtime::input::InputMouseButton::Middle,
    }
}

/// Map a rinch key code string to InputKeyCode.
fn map_key(code: &str) -> Option<rkf_runtime::input::InputKeyCode> {
    use rkf_runtime::input::InputKeyCode::*;
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

    // Track last mouse position for computing deltas.
    let last_mx = std::cell::Cell::new(0.0f32);
    let last_my = std::cell::Cell::new(0.0f32);

    let cmd_tx = cmd.0.clone();
    surface.set_event_handler(move |event| {
        use SurfaceEvent::*;
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
                if let Some(key) = map_key(&key_data.code) {
                    let _ = cmd_tx.send(rkp_engine::EngineCommand::KeyDown { key });
                }
            }
            KeyUp(key_data) => {
                if let Some(key) = map_key(&key_data.code) {
                    let _ = cmd_tx.send(rkp_engine::EngineCommand::KeyUp { key });
                }
            }
            _ => {}
        }
    });

    rsx! {
        RenderSurface { surface: Some(surface.clone()) }
    }
}
