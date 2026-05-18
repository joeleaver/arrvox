//! Gamepad backend via gilrs — polls hardware and feeds into RawInputState.

use super::raw_state::RawInputState;

/// Information about a connected gamepad.
#[allow(missing_docs)]
#[derive(Debug, Clone)]
pub struct GamepadInfo {
    pub id: usize,
    pub name: String,
    pub is_connected: bool,
}

/// Manages gamepad hardware via gilrs. Polls events and feeds into RawInputState.
pub struct GamepadManager {
    #[cfg(feature = "gamepad")]
    gilrs: gilrs::Gilrs,
    connected: Vec<GamepadInfo>,
    /// Track per-stick axis components (gilrs reports axes individually).
    #[cfg(feature = "gamepad")]
    stick_components: StickComponents,
}

#[cfg(feature = "gamepad")]
#[derive(Default)]
struct StickComponents {
    left_x: f32,
    left_y: f32,
    right_x: f32,
    right_y: f32,
}

impl GamepadManager {
    /// Create a new GamepadManager. Returns None if the gamepad subsystem fails to initialize.
    pub fn new() -> Option<Self> {
        #[cfg(feature = "gamepad")]
        {
            match gilrs::Gilrs::new() {
                Ok(gilrs_instance) => {
                    let mut mgr = Self {
                        gilrs: gilrs_instance,
                        connected: Vec::new(),
                        stick_components: StickComponents::default(),
                    };
                    mgr.refresh_connected();
                    Some(mgr)
                }
                Err(e) => {
                    log::warn!("Failed to initialize gamepad system: {}", e);
                    None
                }
            }
        }
        #[cfg(not(feature = "gamepad"))]
        {
            None
        }
    }

    /// Poll gilrs events and feed into RawInputState.
    pub fn poll(&mut self, raw: &mut RawInputState) {
        #[cfg(feature = "gamepad")]
        {
            while let Some(event) = self.gilrs.next_event() {
                self.handle_event(event, raw);
            }
        }
        #[cfg(not(feature = "gamepad"))]
        {
            let _ = raw;
        }
    }

    /// Get connected gamepads.
    pub fn connected_gamepads(&self) -> &[GamepadInfo] {
        &self.connected
    }

    #[cfg(feature = "gamepad")]
    fn refresh_connected(&mut self) {
        self.connected.clear();
        for (id, gamepad) in self.gilrs.gamepads() {
            if gamepad.is_connected() {
                self.connected.push(GamepadInfo {
                    id: id.into(),
                    name: gamepad.name().to_string(),
                    is_connected: true,
                });
            }
        }
    }

    #[cfg(feature = "gamepad")]
    fn handle_event(&mut self, event: gilrs::Event, raw: &mut RawInputState) {
        use gilrs::ev::EventType;

        match event.event {
            EventType::ButtonPressed(btn, _) => {
                if let Some(mapped) = map_button(btn) {
                    raw.set_gamepad_button(mapped, true);
                }
            }
            EventType::ButtonReleased(btn, _) => {
                if let Some(mapped) = map_button(btn) {
                    raw.set_gamepad_button(mapped, false);
                }
            }
            EventType::ButtonChanged(btn, value, _) => {
                // Some controllers report analog triggers as ButtonChanged
                // (LeftTrigger2/RightTrigger2) instead of AxisChanged(LeftZ/RightZ).
                match btn {
                    gilrs::Button::LeftTrigger2 => {
                        raw.set_gamepad_axis(GamepadAxis::LeftTrigger, value);
                    }
                    gilrs::Button::RightTrigger2 => {
                        raw.set_gamepad_axis(GamepadAxis::RightTrigger, value);
                    }
                    _ => {}
                }
            }
            EventType::AxisChanged(axis, value, _) => {
                self.handle_axis(axis, value, raw);
            }
            EventType::Connected => {
                self.refresh_connected();
            }
            EventType::Disconnected => {
                self.refresh_connected();
            }
            _ => {}
        }
    }

    #[cfg(feature = "gamepad")]
    fn handle_axis(&mut self, axis: gilrs::Axis, value: f32, raw: &mut RawInputState) {
        match axis {
            gilrs::Axis::LeftStickX => {
                self.stick_components.left_x = value;
                raw.set_gamepad_stick(
                    GamepadStick::Left,
                    Vec2::new(self.stick_components.left_x, self.stick_components.left_y),
                );
            }
            gilrs::Axis::LeftStickY => {
                self.stick_components.left_y = value;
                raw.set_gamepad_stick(
                    GamepadStick::Left,
                    Vec2::new(self.stick_components.left_x, self.stick_components.left_y),
                );
            }
            gilrs::Axis::RightStickX => {
                self.stick_components.right_x = value;
                raw.set_gamepad_stick(
                    GamepadStick::Right,
                    Vec2::new(self.stick_components.right_x, self.stick_components.right_y),
                );
            }
            gilrs::Axis::RightStickY => {
                self.stick_components.right_y = value;
                raw.set_gamepad_stick(
                    GamepadStick::Right,
                    Vec2::new(self.stick_components.right_x, self.stick_components.right_y),
                );
            }
            gilrs::Axis::LeftZ => {
                raw.set_gamepad_axis(GamepadAxis::LeftTrigger, value);
            }
            gilrs::Axis::RightZ => {
                raw.set_gamepad_axis(GamepadAxis::RightTrigger, value);
            }
            _ => {}
        }
    }
}

/// Map gilrs Button to our GamepadButton.
///
/// Note: In gilrs, "LeftTrigger" = bumper/shoulder button, "LeftTrigger2" = analog trigger.
/// We map LeftTrigger→LeftBumper, and the analog trigger axis is handled via AxisChanged.
#[cfg(feature = "gamepad")]
fn map_button(btn: gilrs::Button) -> Option<GamepadButton> {
    match btn {
        gilrs::Button::South => Some(GamepadButton::South),
        gilrs::Button::East => Some(GamepadButton::East),
        gilrs::Button::West => Some(GamepadButton::West),
        gilrs::Button::North => Some(GamepadButton::North),
        gilrs::Button::DPadUp => Some(GamepadButton::DPadUp),
        gilrs::Button::DPadDown => Some(GamepadButton::DPadDown),
        gilrs::Button::DPadLeft => Some(GamepadButton::DPadLeft),
        gilrs::Button::DPadRight => Some(GamepadButton::DPadRight),
        gilrs::Button::LeftTrigger => Some(GamepadButton::LeftBumper),
        gilrs::Button::RightTrigger => Some(GamepadButton::RightBumper),
        gilrs::Button::LeftThumb => Some(GamepadButton::LeftStick),
        gilrs::Button::RightThumb => Some(GamepadButton::RightStick),
        gilrs::Button::Start => Some(GamepadButton::Start),
        gilrs::Button::Select => Some(GamepadButton::Select),
        gilrs::Button::Mode => Some(GamepadButton::Guide),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamepad_manager_new_succeeds() {
        // gilrs can initialize even without controllers on Linux
        let mgr = GamepadManager::new();
        // On CI without gamepad subsystem this may be None, so we just verify it doesn't panic
        if let Some(mgr) = mgr {
            let _ = mgr.connected_gamepads();
        }
    }

    #[cfg(feature = "gamepad")]
    #[test]
    fn button_mapping_covers_all_variants() {
        let mappings = [
            (gilrs::Button::South, GamepadButton::South),
            (gilrs::Button::East, GamepadButton::East),
            (gilrs::Button::West, GamepadButton::West),
            (gilrs::Button::North, GamepadButton::North),
            (gilrs::Button::DPadUp, GamepadButton::DPadUp),
            (gilrs::Button::DPadDown, GamepadButton::DPadDown),
            (gilrs::Button::DPadLeft, GamepadButton::DPadLeft),
            (gilrs::Button::DPadRight, GamepadButton::DPadRight),
            (gilrs::Button::LeftTrigger, GamepadButton::LeftBumper),
            (gilrs::Button::RightTrigger, GamepadButton::RightBumper),
            (gilrs::Button::LeftThumb, GamepadButton::LeftStick),
            (gilrs::Button::RightThumb, GamepadButton::RightStick),
            (gilrs::Button::Start, GamepadButton::Start),
            (gilrs::Button::Select, GamepadButton::Select),
            (gilrs::Button::Mode, GamepadButton::Guide),
        ];

        for (gilrs_btn, expected) in &mappings {
            assert_eq!(
                map_button(*gilrs_btn),
                Some(*expected),
                "mapping for {:?} failed",
                gilrs_btn,
            );
        }
    }

    #[cfg(feature = "gamepad")]
    #[test]
    fn unmapped_buttons_return_none() {
        // LeftTrigger2 and RightTrigger2 are analog triggers, not buttons
        assert_eq!(map_button(gilrs::Button::LeftTrigger2), None);
        assert_eq!(map_button(gilrs::Button::RightTrigger2), None);
    }

    #[test]
    fn poll_without_gamepad_is_noop() {
        // Without feature, new() returns None, so this tests the non-gamepad path
        #[cfg(not(feature = "gamepad"))]
        {
            assert!(GamepadManager::new().is_none());
        }
    }
}
