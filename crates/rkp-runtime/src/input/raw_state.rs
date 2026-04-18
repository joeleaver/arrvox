//! Raw physical input state — tracks current and per-frame key/mouse/gamepad state.

use glam::Vec2;
use std::collections::{HashMap, HashSet};

use super::types::{GamepadAxis, GamepadButton, GamepadStick, InputKeyCode, InputMouseButton, PhysicalInput};

/// Tracks raw physical input state for the current frame.
#[derive(Debug, Clone)]
pub struct RawInputState {
    pub(crate) keys_pressed: HashSet<InputKeyCode>,
    pub(crate) keys_just_pressed: HashSet<InputKeyCode>,
    pub(crate) keys_just_released: HashSet<InputKeyCode>,
    pub(crate) mouse_buttons: [bool; 3],
    pub(crate) mouse_buttons_just_pressed: [bool; 3],
    pub(crate) mouse_buttons_just_released: [bool; 3],
    pub(crate) mouse_delta: Vec2,
    pub(crate) scroll_delta: f32,
    pub(crate) gamepad_buttons: HashSet<GamepadButton>,
    pub(crate) gamepad_buttons_just_pressed: HashSet<GamepadButton>,
    pub(crate) gamepad_buttons_just_released: HashSet<GamepadButton>,
    pub(crate) gamepad_axes: HashMap<GamepadAxis, f32>,
    pub(crate) gamepad_sticks: HashMap<GamepadStick, Vec2>,
}

impl Default for RawInputState {
    fn default() -> Self {
        Self {
            keys_pressed: HashSet::new(),
            keys_just_pressed: HashSet::new(),
            keys_just_released: HashSet::new(),
            mouse_buttons: [false; 3],
            mouse_buttons_just_pressed: [false; 3],
            mouse_buttons_just_released: [false; 3],
            mouse_delta: Vec2::ZERO,
            scroll_delta: 0.0,
            gamepad_buttons: HashSet::new(),
            gamepad_buttons_just_pressed: HashSet::new(),
            gamepad_buttons_just_released: HashSet::new(),
            gamepad_axes: HashMap::new(),
            gamepad_sticks: HashMap::new(),
        }
    }
}

impl RawInputState {
    /// Create a new empty raw input state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear per-frame transient state. Call at the start of each frame.
    pub fn begin_frame(&mut self) {
        self.keys_just_pressed.clear();
        self.keys_just_released.clear();
        self.mouse_buttons_just_pressed = [false; 3];
        self.mouse_buttons_just_released = [false; 3];
        self.mouse_delta = Vec2::ZERO;
        self.scroll_delta = 0.0;
        self.gamepad_buttons_just_pressed.clear();
        self.gamepad_buttons_just_released.clear();
    }

    /// Record a key press.
    pub fn key_down(&mut self, key: InputKeyCode) {
        if self.keys_pressed.insert(key) {
            self.keys_just_pressed.insert(key);
        }
    }

    /// Record a key release.
    pub fn key_up(&mut self, key: InputKeyCode) {
        if self.keys_pressed.remove(&key) {
            self.keys_just_released.insert(key);
        }
    }

    /// Record a mouse button press.
    pub fn mouse_button_down(&mut self, button: InputMouseButton) {
        let idx = mouse_button_index(button);
        if !self.mouse_buttons[idx] {
            self.mouse_buttons_just_pressed[idx] = true;
        }
        self.mouse_buttons[idx] = true;
    }

    /// Record a mouse button release.
    pub fn mouse_button_up(&mut self, button: InputMouseButton) {
        let idx = mouse_button_index(button);
        if self.mouse_buttons[idx] {
            self.mouse_buttons_just_released[idx] = true;
        }
        self.mouse_buttons[idx] = false;
    }

    /// Accumulate mouse movement delta.
    pub fn add_mouse_delta(&mut self, delta: Vec2) {
        self.mouse_delta += delta;
    }

    /// Accumulate scroll wheel delta.
    pub fn add_scroll(&mut self, delta: f32) {
        self.scroll_delta += delta;
    }

    /// Set a gamepad button state.
    pub fn set_gamepad_button(&mut self, button: GamepadButton, pressed: bool) {
        if pressed {
            if self.gamepad_buttons.insert(button) {
                self.gamepad_buttons_just_pressed.insert(button);
            }
        } else if self.gamepad_buttons.remove(&button) {
            self.gamepad_buttons_just_released.insert(button);
        }
    }

    /// Set a gamepad axis value.
    pub fn set_gamepad_axis(&mut self, axis: GamepadAxis, value: f32) {
        self.gamepad_axes.insert(axis, value);
    }

    /// Set a gamepad stick value.
    pub fn set_gamepad_stick(&mut self, stick: GamepadStick, value: Vec2) {
        self.gamepad_sticks.insert(stick, value);
    }

    /// Check if a key is currently pressed.
    pub fn is_key_pressed(&self, key: InputKeyCode) -> bool {
        self.keys_pressed.contains(&key)
    }

    /// Check if a key was just pressed this frame.
    pub fn is_key_just_pressed(&self, key: InputKeyCode) -> bool {
        self.keys_just_pressed.contains(&key)
    }

    /// Check if a key was just released this frame.
    pub fn is_key_just_released(&self, key: InputKeyCode) -> bool {
        self.keys_just_released.contains(&key)
    }

    /// Check if a mouse button is currently pressed.
    pub fn is_mouse_button_pressed(&self, button: InputMouseButton) -> bool {
        self.mouse_buttons[mouse_button_index(button)]
    }

    /// Check if a mouse button was just pressed this frame.
    pub fn is_mouse_button_just_pressed(&self, button: InputMouseButton) -> bool {
        self.mouse_buttons_just_pressed[mouse_button_index(button)]
    }

    /// Check if a gamepad button is currently pressed.
    pub fn is_gamepad_button_pressed(&self, button: GamepadButton) -> bool {
        self.gamepad_buttons.contains(&button)
    }

    /// All currently pressed gamepad buttons.
    pub fn gamepad_buttons_pressed(&self) -> &HashSet<GamepadButton> {
        &self.gamepad_buttons
    }

    /// All gamepad axis values (triggers).
    pub fn gamepad_axes_values(&self) -> &HashMap<GamepadAxis, f32> {
        &self.gamepad_axes
    }

    /// All gamepad stick values.
    pub fn gamepad_sticks_values(&self) -> &HashMap<GamepadStick, Vec2> {
        &self.gamepad_sticks
    }

    /// Check if a physical input is active (digital: pressed, analog: non-zero).
    pub fn is_physical_input_active(&self, input: &PhysicalInput) -> bool {
        match input {
            PhysicalInput::Key(k) => self.is_key_pressed(*k),
            PhysicalInput::MouseButton(b) => self.is_mouse_button_pressed(*b),
            PhysicalInput::MouseDelta => self.mouse_delta != Vec2::ZERO,
            PhysicalInput::ScrollWheel => self.scroll_delta != 0.0,
            PhysicalInput::GamepadButton(b) => self.is_gamepad_button_pressed(*b),
            PhysicalInput::GamepadAxis(a) => {
                self.gamepad_axes.get(a).copied().unwrap_or(0.0) != 0.0
            }
            PhysicalInput::GamepadStick(s) => {
                self.gamepad_sticks.get(s).copied().unwrap_or(Vec2::ZERO) != Vec2::ZERO
            }
        }
    }

    /// Get the value of a physical input as (digital, axis_1d, axis_2d).
    pub fn physical_input_value(&self, input: &PhysicalInput) -> (bool, f32, Vec2) {
        match input {
            PhysicalInput::Key(k) => {
                let pressed = self.is_key_pressed(*k);
                (pressed, if pressed { 1.0 } else { 0.0 }, Vec2::ZERO)
            }
            PhysicalInput::MouseButton(b) => {
                let pressed = self.is_mouse_button_pressed(*b);
                (pressed, if pressed { 1.0 } else { 0.0 }, Vec2::ZERO)
            }
            PhysicalInput::MouseDelta => {
                let active = self.mouse_delta != Vec2::ZERO;
                (active, self.mouse_delta.length(), self.mouse_delta)
            }
            PhysicalInput::ScrollWheel => {
                let active = self.scroll_delta != 0.0;
                (active, self.scroll_delta, Vec2::ZERO)
            }
            PhysicalInput::GamepadButton(b) => {
                let pressed = self.is_gamepad_button_pressed(*b);
                (pressed, if pressed { 1.0 } else { 0.0 }, Vec2::ZERO)
            }
            PhysicalInput::GamepadAxis(a) => {
                let val = self.gamepad_axes.get(a).copied().unwrap_or(0.0);
                (val != 0.0, val, Vec2::ZERO)
            }
            PhysicalInput::GamepadStick(s) => {
                let val = self.gamepad_sticks.get(s).copied().unwrap_or(Vec2::ZERO);
                (val != Vec2::ZERO, val.length(), val)
            }
        }
    }
}

fn mouse_button_index(button: InputMouseButton) -> usize {
    match button {
        InputMouseButton::Left => 0,
        InputMouseButton::Right => 1,
        InputMouseButton::Middle => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_press_release_lifecycle() {
        let mut raw = RawInputState::new();

        raw.key_down(InputKeyCode::W);
        assert!(raw.is_key_pressed(InputKeyCode::W));
        assert!(raw.is_key_just_pressed(InputKeyCode::W));

        // Next frame: just_pressed clears
        raw.begin_frame();
        assert!(raw.is_key_pressed(InputKeyCode::W));
        assert!(!raw.is_key_just_pressed(InputKeyCode::W));

        // Release
        raw.key_up(InputKeyCode::W);
        assert!(!raw.is_key_pressed(InputKeyCode::W));
        assert!(raw.is_key_just_released(InputKeyCode::W));

        // Next frame: just_released clears
        raw.begin_frame();
        assert!(!raw.is_key_just_released(InputKeyCode::W));
    }

    #[test]
    fn duplicate_key_down_no_double_just_pressed() {
        let mut raw = RawInputState::new();
        raw.key_down(InputKeyCode::A);
        raw.begin_frame();
        // Key still held, press again — should not re-trigger just_pressed
        raw.key_down(InputKeyCode::A);
        assert!(raw.is_key_pressed(InputKeyCode::A));
        assert!(!raw.is_key_just_pressed(InputKeyCode::A));
    }

    #[test]
    fn mouse_delta_accumulates_and_clears() {
        let mut raw = RawInputState::new();
        raw.add_mouse_delta(Vec2::new(10.0, 5.0));
        raw.add_mouse_delta(Vec2::new(-3.0, 2.0));
        assert_eq!(raw.mouse_delta, Vec2::new(7.0, 7.0));

        raw.begin_frame();
        assert_eq!(raw.mouse_delta, Vec2::ZERO);
    }

    #[test]
    fn scroll_accumulates_and_clears() {
        let mut raw = RawInputState::new();
        raw.add_scroll(1.0);
        raw.add_scroll(-0.5);
        assert_eq!(raw.scroll_delta, 0.5);

        raw.begin_frame();
        assert_eq!(raw.scroll_delta, 0.0);
    }

    #[test]
    fn mouse_button_press_release() {
        let mut raw = RawInputState::new();
        raw.mouse_button_down(InputMouseButton::Left);
        assert!(raw.is_mouse_button_pressed(InputMouseButton::Left));
        assert!(raw.is_mouse_button_just_pressed(InputMouseButton::Left));
        assert!(!raw.is_mouse_button_pressed(InputMouseButton::Right));

        raw.begin_frame();
        assert!(raw.is_mouse_button_pressed(InputMouseButton::Left));
        assert!(!raw.is_mouse_button_just_pressed(InputMouseButton::Left));

        raw.mouse_button_up(InputMouseButton::Left);
        assert!(!raw.is_mouse_button_pressed(InputMouseButton::Left));
    }

    #[test]
    fn gamepad_button_state() {
        let mut raw = RawInputState::new();
        raw.set_gamepad_button(GamepadButton::South, true);
        assert!(raw.is_gamepad_button_pressed(GamepadButton::South));
        assert!(!raw.is_gamepad_button_pressed(GamepadButton::East));

        raw.set_gamepad_button(GamepadButton::South, false);
        assert!(!raw.is_gamepad_button_pressed(GamepadButton::South));
    }

    #[test]
    fn gamepad_axis_and_stick() {
        let mut raw = RawInputState::new();
        raw.set_gamepad_axis(GamepadAxis::LeftTrigger, 0.75);
        raw.set_gamepad_stick(GamepadStick::Left, Vec2::new(0.5, -0.3));

        let (active, val, _) = raw.physical_input_value(&PhysicalInput::GamepadAxis(GamepadAxis::LeftTrigger));
        assert!(active);
        assert_eq!(val, 0.75);

        let (active, _, vec) = raw.physical_input_value(&PhysicalInput::GamepadStick(GamepadStick::Left));
        assert!(active);
        assert_eq!(vec, Vec2::new(0.5, -0.3));
    }

    #[test]
    fn physical_input_active_checks() {
        let mut raw = RawInputState::new();

        assert!(!raw.is_physical_input_active(&PhysicalInput::Key(InputKeyCode::Space)));
        raw.key_down(InputKeyCode::Space);
        assert!(raw.is_physical_input_active(&PhysicalInput::Key(InputKeyCode::Space)));

        assert!(!raw.is_physical_input_active(&PhysicalInput::MouseDelta));
        raw.add_mouse_delta(Vec2::new(1.0, 0.0));
        assert!(raw.is_physical_input_active(&PhysicalInput::MouseDelta));

        assert!(!raw.is_physical_input_active(&PhysicalInput::ScrollWheel));
        raw.add_scroll(1.0);
        assert!(raw.is_physical_input_active(&PhysicalInput::ScrollWheel));
    }

    #[test]
    fn physical_input_value_key() {
        let mut raw = RawInputState::new();
        let (d, a, v) = raw.physical_input_value(&PhysicalInput::Key(InputKeyCode::W));
        assert!(!d);
        assert_eq!(a, 0.0);
        assert_eq!(v, Vec2::ZERO);

        raw.key_down(InputKeyCode::W);
        let (d, a, _) = raw.physical_input_value(&PhysicalInput::Key(InputKeyCode::W));
        assert!(d);
        assert_eq!(a, 1.0);
    }
}
