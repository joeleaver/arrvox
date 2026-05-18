//! InputSystem — main public API for the input system.

use glam::Vec2;
use std::collections::HashMap;

use super::action_map::ActionMap;
use super::binding::Binding;
use super::gamepad::{GamepadInfo, GamepadManager};
use super::gamepad_ui::{self, GamepadUiEvent};
use super::raw_state::RawInputState;
use super::types::*;

mod evaluate;
mod binding;

#[cfg(test)]
mod tests;

/// The main input system. Manages action maps, raw input, and action evaluation.
pub struct InputSystem {
    maps: Vec<ActionMap>,
    active_map_index: Option<usize>,
    raw: RawInputState,
    action_states: HashMap<String, ActionState>,
    dead_zone: f32,
    gamepad_manager: Option<GamepadManager>,
}

impl std::fmt::Debug for InputSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InputSystem")
            .field("maps", &self.maps.len())
            .field("active_map_index", &self.active_map_index)
            .field("action_states", &self.action_states.len())
            .field("dead_zone", &self.dead_zone)
            .finish()
    }
}

impl InputSystem {
    /// Create a new input system with default dead zone (0.1).
    pub fn new() -> Self {
        Self {
            maps: Vec::new(),
            active_map_index: None,
            raw: RawInputState::new(),
            action_states: HashMap::new(),
            dead_zone: 0.1,
            gamepad_manager: None,
        }
    }

    /// Create a new input system with a custom global dead zone.
    pub fn with_dead_zone(dead_zone: f32) -> Self {
        Self {
            dead_zone,
            ..Self::new()
        }
    }

    /// Add an action map.
    pub fn add_map(&mut self, map: ActionMap) {
        self.maps.push(map);
    }

    /// Set the active action map by name. Returns false if not found.
    pub fn set_active_map(&mut self, name: &str) -> bool {
        if let Some(idx) = self.maps.iter().position(|m| m.name == name) {
            self.active_map_index = Some(idx);
            self.action_states.clear();
            true
        } else {
            false
        }
    }

    /// Get the name of the active action map.
    pub fn active_map(&self) -> Option<&str> {
        self.active_map_index.map(|i| self.maps[i].name.as_str())
    }

    /// Begin a new frame — clears per-frame raw input state.
    pub fn begin_frame(&mut self) {
        self.raw.begin_frame();
    }

    /// Reset all input state (all keys/buttons released, all actions waiting).
    /// Call on focus loss to prevent stuck keys/buttons.
    pub fn reset(&mut self) {
        self.raw = RawInputState::new();
        self.action_states.clear();
    }

    // --- Feed raw input ---

    /// Feed a key press event.
    pub fn feed_key_down(&mut self, key: InputKeyCode) {
        self.raw.key_down(key);
    }

    /// Feed a key release event.
    pub fn feed_key_up(&mut self, key: InputKeyCode) {
        self.raw.key_up(key);
    }

    /// Feed a mouse button event.
    pub fn feed_mouse_button(&mut self, button: InputMouseButton, pressed: bool) {
        if pressed {
            self.raw.mouse_button_down(button);
        } else {
            self.raw.mouse_button_up(button);
        }
    }

    /// Feed mouse movement delta.
    pub fn feed_mouse_delta(&mut self, delta: Vec2) {
        self.raw.add_mouse_delta(delta);
    }

    /// Feed scroll wheel delta.
    pub fn feed_scroll(&mut self, delta: f32) {
        self.raw.add_scroll(delta);
    }

    /// Feed a gamepad button event.
    pub fn feed_gamepad_button(&mut self, button: GamepadButton, pressed: bool) {
        self.raw.set_gamepad_button(button, pressed);
    }

    /// Feed a gamepad axis value.
    pub fn feed_gamepad_axis(&mut self, axis: GamepadAxis, value: f32) {
        self.raw.set_gamepad_axis(axis, value);
    }

    /// Feed a gamepad stick value.
    pub fn feed_gamepad_stick(&mut self, stick: GamepadStick, value: Vec2) {
        self.raw.set_gamepad_stick(stick, value);
    }


    // --- Query actions ---

    /// Get the state of an action by name.
    pub fn action(&self, name: &str) -> Option<&ActionState> {
        self.action_states.get(name)
    }

    /// Check if an action is currently active (digital).
    pub fn pressed(&self, name: &str) -> bool {
        self.action_states.get(name).is_some_and(|s| s.digital)
    }

    /// Check if an action was just started this frame.
    pub fn just_pressed(&self, name: &str) -> bool {
        self.action_states
            .get(name)
            .is_some_and(|s| s.started_this_frame)
    }

    /// Check if an action was just canceled this frame.
    pub fn just_released(&self, name: &str) -> bool {
        self.action_states
            .get(name)
            .is_some_and(|s| s.canceled_this_frame)
    }

    /// Get the 1D axis value of an action.
    pub fn axis_1d(&self, name: &str) -> f32 {
        self.action_states.get(name).map_or(0.0, |s| s.axis_1d)
    }

    /// Get the 2D axis value of an action.
    pub fn axis_2d(&self, name: &str) -> Vec2 {
        self.action_states
            .get(name)
            .map_or(Vec2::ZERO, |s| s.axis_2d)
    }

    /// Get read-only access to raw input state.
    pub fn raw_state(&self) -> &RawInputState {
        &self.raw
    }

    // --- Gamepad ---

    /// Enable gamepad support. Returns true if the gamepad subsystem initialized.
    pub fn enable_gamepad(&mut self) -> bool {
        if self.gamepad_manager.is_some() {
            return true;
        }
        match GamepadManager::new() {
            Some(mgr) => {
                self.gamepad_manager = Some(mgr);
                true
            }
            None => false,
        }
    }

    /// Poll connected gamepads and feed events into raw input state.
    pub fn poll_gamepads(&mut self) {
        if let Some(ref mut mgr) = self.gamepad_manager {
            mgr.poll(&mut self.raw);
        }
    }

    /// Get currently connected gamepads.
    pub fn connected_gamepads(&self) -> &[GamepadInfo] {
        self.gamepad_manager
            .as_ref()
            .map_or(&[], |m| m.connected_gamepads())
    }

    /// Get UI navigation events from gamepad state (call after poll_gamepads + begin_frame).
    pub fn gamepad_ui_events(&self) -> Vec<GamepadUiEvent> {
        gamepad_ui::translate_gamepad_for_ui(
            &self.raw.gamepad_buttons_just_pressed,
            &self.raw.gamepad_buttons_just_released,
        )
    }

}

/// Collect all physical inputs referenced by a binding.
fn collect_inputs(binding: &Binding) -> Vec<PhysicalInput> {
    match binding {
        Binding::Simple { input, .. } => vec![*input],
        Binding::Composite2D { up, down, left, right } => vec![*up, *down, *left, *right],
        Binding::CompositeAxis { positive, negative } => vec![*positive, *negative],
    }
}

impl Default for InputSystem {
    fn default() -> Self {
        Self::new()
    }
}

