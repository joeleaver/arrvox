//! Gamepad-to-UI navigation translation layer.
//!
//! Translates gamepad button presses into synthetic keyboard events
//! for UI navigation (DPad → arrows, South → Enter, East → Escape).

use std::collections::HashSet;

use super::types::{GamepadButton, InputKeyCode};

/// A synthetic keyboard event generated from gamepad input for UI navigation.
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GamepadUiEvent {
    pub key: InputKeyCode,
    pub pressed: bool,
}

/// Mapping from gamepad buttons to keyboard keys for UI navigation.
const UI_BUTTON_MAP: &[(GamepadButton, InputKeyCode)] = &[
    (GamepadButton::DPadUp, InputKeyCode::ArrowUp),
    (GamepadButton::DPadDown, InputKeyCode::ArrowDown),
    (GamepadButton::DPadLeft, InputKeyCode::ArrowLeft),
    (GamepadButton::DPadRight, InputKeyCode::ArrowRight),
    (GamepadButton::South, InputKeyCode::Enter),
    (GamepadButton::East, InputKeyCode::Escape),
];

/// Translate gamepad button state changes to UI navigation events.
///
/// Call each frame with the sets of buttons that were just pressed/released.
/// Returns synthetic keyboard events suitable for feeding into UI input handling.
pub fn translate_gamepad_for_ui(
    buttons_just_pressed: &HashSet<GamepadButton>,
    buttons_just_released: &HashSet<GamepadButton>,
) -> Vec<GamepadUiEvent> {
    let mut events = Vec::new();

    for (gp_btn, key) in UI_BUTTON_MAP {
        if buttons_just_pressed.contains(gp_btn) {
            events.push(GamepadUiEvent {
                key: *key,
                pressed: true,
            });
        }
        if buttons_just_released.contains(gp_btn) {
            events.push(GamepadUiEvent {
                key: *key,
                pressed: false,
            });
        }
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dpad_up_pressed_produces_arrow_up() {
        let mut pressed = HashSet::new();
        pressed.insert(GamepadButton::DPadUp);
        let events = translate_gamepad_for_ui(&pressed, &HashSet::new());
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].key, InputKeyCode::ArrowUp);
        assert!(events[0].pressed);
    }

    #[test]
    fn south_pressed_produces_enter() {
        let mut pressed = HashSet::new();
        pressed.insert(GamepadButton::South);
        let events = translate_gamepad_for_ui(&pressed, &HashSet::new());
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].key, InputKeyCode::Enter);
        assert!(events[0].pressed);
    }

    #[test]
    fn east_pressed_produces_escape() {
        let mut pressed = HashSet::new();
        pressed.insert(GamepadButton::East);
        let events = translate_gamepad_for_ui(&pressed, &HashSet::new());
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].key, InputKeyCode::Escape);
        assert!(events[0].pressed);
    }

    #[test]
    fn no_buttons_produces_empty() {
        let events = translate_gamepad_for_ui(&HashSet::new(), &HashSet::new());
        assert!(events.is_empty());
    }

    #[test]
    fn released_buttons_produce_released_events() {
        let mut released = HashSet::new();
        released.insert(GamepadButton::DPadDown);
        let events = translate_gamepad_for_ui(&HashSet::new(), &released);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].key, InputKeyCode::ArrowDown);
        assert!(!events[0].pressed);
    }

    #[test]
    fn multiple_buttons_produce_multiple_events() {
        let mut pressed = HashSet::new();
        pressed.insert(GamepadButton::DPadLeft);
        pressed.insert(GamepadButton::South);
        let events = translate_gamepad_for_ui(&pressed, &HashSet::new());
        assert_eq!(events.len(), 2);
        let keys: HashSet<InputKeyCode> = events.iter().map(|e| e.key).collect();
        assert!(keys.contains(&InputKeyCode::ArrowLeft));
        assert!(keys.contains(&InputKeyCode::Enter));
    }

    #[test]
    fn unmapped_buttons_produce_no_events() {
        let mut pressed = HashSet::new();
        pressed.insert(GamepadButton::LeftBumper);
        pressed.insert(GamepadButton::Start);
        let events = translate_gamepad_for_ui(&pressed, &HashSet::new());
        assert!(events.is_empty());
    }

    #[test]
    fn press_and_release_same_frame() {
        let mut pressed = HashSet::new();
        pressed.insert(GamepadButton::DPadRight);
        let mut released = HashSet::new();
        released.insert(GamepadButton::DPadLeft);
        let events = translate_gamepad_for_ui(&pressed, &released);
        assert_eq!(events.len(), 2);
        let press_event = events.iter().find(|e| e.pressed).unwrap();
        let release_event = events.iter().find(|e| !e.pressed).unwrap();
        assert_eq!(press_event.key, InputKeyCode::ArrowRight);
        assert_eq!(release_event.key, InputKeyCode::ArrowLeft);
    }
}
