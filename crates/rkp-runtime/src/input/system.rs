//! InputSystem — main public API for the input system.

use glam::Vec2;
use std::collections::HashMap;

use super::action_map::ActionMap;
use super::binding::Binding;
use super::gamepad::{GamepadInfo, GamepadManager};
use super::gamepad_ui::{self, GamepadUiEvent};
use super::raw_state::RawInputState;
use super::types::*;

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

    /// Evaluate all bindings in the active action map and update action states.
    pub fn evaluate(&mut self) {
        let map_index = match self.active_map_index {
            Some(i) => i,
            None => return,
        };

        // Collect action evaluation data first to avoid borrow conflicts
        let action_evals: Vec<(String, bool, f32, Vec2, f32)> = self.maps[map_index]
            .actions
            .iter()
            .map(|action| {
                let mut combined_digital = false;
                let mut combined_1d = 0.0f32;
                let mut combined_2d = Vec2::ZERO;
                let mut max_mag = 0.0f32;

                for binding in &action.bindings {
                    let (d, a1, a2) = binding.evaluate(&self.raw);
                    combined_digital |= d;

                    if a1.abs() > combined_1d.abs() {
                        combined_1d = a1;
                    }

                    let mag = a2.length();
                    if mag > max_mag {
                        max_mag = mag;
                        combined_2d = a2;
                    }
                }

                let effective_dead_zone = if action.dead_zone > 0.0 {
                    action.dead_zone
                } else {
                    self.dead_zone
                };

                (
                    action.name.clone(),
                    combined_digital,
                    combined_1d,
                    combined_2d,
                    effective_dead_zone,
                )
            })
            .collect();

        for (name, digital, axis_1d, axis_2d, dead_zone) in action_evals {
            // Apply dead zone to analog values
            let axis_1d = if axis_1d.abs() < dead_zone { 0.0 } else { axis_1d };
            let axis_2d = if axis_2d.length() < dead_zone { Vec2::ZERO } else { axis_2d };

            let is_active = digital || axis_1d != 0.0 || axis_2d != Vec2::ZERO;

            let prev_state = self.action_states.get(&name);
            let prev_phase = prev_state
                .map(|s| s.phase)
                .unwrap_or(ActionPhase::Waiting);

            let (new_phase, started, performed, canceled) = match (prev_phase, is_active) {
                (ActionPhase::Waiting, true) => (ActionPhase::Started, true, false, false),
                (ActionPhase::Waiting, false) => (ActionPhase::Waiting, false, false, false),
                (ActionPhase::Started, true) => (ActionPhase::Performed, false, true, false),
                (ActionPhase::Started, false) => (ActionPhase::Canceled, false, false, true),
                (ActionPhase::Performed, true) => (ActionPhase::Performed, false, false, false),
                (ActionPhase::Performed, false) => (ActionPhase::Canceled, false, false, true),
                (ActionPhase::Canceled, true) => (ActionPhase::Started, true, false, false),
                (ActionPhase::Canceled, false) => (ActionPhase::Waiting, false, false, false),
            };

            self.action_states.insert(name, ActionState {
                phase: new_phase,
                digital: is_active,
                axis_1d,
                axis_2d,
                started_this_frame: started,
                performed_this_frame: performed,
                canceled_this_frame: canceled,
            });
        }
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

    // --- Rebinding API ---

    /// Get the active action map (mutable).
    fn active_map_mut(&mut self) -> Option<&mut ActionMap> {
        self.active_map_index.map(|i| &mut self.maps[i])
    }

    /// Replace a binding at the given index for an action in the active map.
    pub fn set_binding(&mut self, action: &str, index: usize, binding: Binding) -> bool {
        if let Some(map) = self.active_map_mut() {
            if let Some(act) = map.find_action_mut(action) {
                if index < act.bindings.len() {
                    act.bindings[index] = binding;
                    return true;
                }
            }
        }
        false
    }

    /// Replace a specific part of a composite binding.
    pub fn set_binding_part(
        &mut self,
        action: &str,
        index: usize,
        part: &str,
        input: PhysicalInput,
    ) -> bool {
        if let Some(map) = self.active_map_mut() {
            if let Some(act) = map.find_action_mut(action) {
                if let Some(binding) = act.bindings.get_mut(index) {
                    match binding {
                        Binding::Composite2D {
                            up,
                            down,
                            left,
                            right,
                        } => match part {
                            "up" => { *up = input; return true; }
                            "down" => { *down = input; return true; }
                            "left" => { *left = input; return true; }
                            "right" => { *right = input; return true; }
                            _ => {}
                        },
                        Binding::CompositeAxis {
                            positive,
                            negative,
                        } => match part {
                            "positive" => { *positive = input; return true; }
                            "negative" => { *negative = input; return true; }
                            _ => {}
                        },
                        Binding::Simple { .. } => {}
                    }
                }
            }
        }
        false
    }

    /// Add an additional binding to an action.
    pub fn add_binding(&mut self, action: &str, binding: Binding) -> bool {
        if let Some(map) = self.active_map_mut() {
            if let Some(act) = map.find_action_mut(action) {
                act.bindings.push(binding);
                return true;
            }
        }
        false
    }

    /// Remove a binding from an action.
    pub fn remove_binding(&mut self, action: &str, index: usize) -> bool {
        if let Some(map) = self.active_map_mut() {
            if let Some(act) = map.find_action_mut(action) {
                if index < act.bindings.len() {
                    act.bindings.remove(index);
                    return true;
                }
            }
        }
        false
    }

    /// Get current bindings for an action.
    pub fn get_bindings(&self, action: &str) -> Option<&[Binding]> {
        self.active_map_index
            .and_then(|i| self.maps[i].find_action(action))
            .map(|a| a.bindings.as_slice())
    }

    /// Find actions in the active map that conflict with this binding.
    /// Returns action names whose bindings share any physical input with the given binding.
    pub fn find_conflicts(&self, binding: &Binding) -> Vec<String> {
        let check_inputs = collect_inputs(binding);
        let map = match self.active_map_index {
            Some(i) => &self.maps[i],
            None => return Vec::new(),
        };
        let mut conflicts = Vec::new();
        for act in &map.actions {
            for b in &act.bindings {
                let inputs = collect_inputs(b);
                if inputs.iter().any(|i| check_inputs.contains(i)) {
                    conflicts.push(act.name.clone());
                    break;
                }
            }
        }
        conflicts
    }

    /// Reset one action to its default bindings.
    pub fn reset_binding(&mut self, action: &str) -> bool {
        if let Some(map) = self.active_map_mut() {
            if let Some(act) = map.find_action_mut(action) {
                act.reset_bindings();
                return true;
            }
        }
        false
    }

    /// Extract shortcut display strings for all actions in a named map.
    ///
    /// Returns a map of `action_id → display_string` using the first binding's
    /// `display_string()`. Actions with no bindings are omitted.
    pub fn shortcut_strings(&self, map_name: &str) -> HashMap<String, String> {
        let mut result = HashMap::new();
        if let Some(map) = self.maps.iter().find(|m| m.name == map_name) {
            for action in &map.actions {
                if let Some(first_binding) = action.bindings.first() {
                    result.insert(action.name.clone(), first_binding.display_string());
                }
            }
        }
        result
    }

    /// Reset entire active map to defaults.
    pub fn reset_active_map(&mut self) -> bool {
        if let Some(map) = self.active_map_mut() {
            for act in &mut map.actions {
                act.reset_bindings();
            }
            true
        } else {
            false
        }
    }

    /// Export overrides (delta from defaults) for a specific map.
    pub fn export_overrides(
        &self,
        map_name: &str,
    ) -> Option<HashMap<String, Vec<Binding>>> {
        let map = self.maps.iter().find(|m| m.name == map_name)?;
        let mut overrides = HashMap::new();
        for act in &map.actions {
            if act.bindings != act.default_bindings {
                overrides.insert(act.name.clone(), act.bindings.clone());
            }
        }
        Some(overrides)
    }

    /// Apply overrides to a map.
    pub fn apply_overrides(
        &mut self,
        map_name: &str,
        overrides: &HashMap<String, Vec<Binding>>,
    ) -> bool {
        let map = match self.maps.iter_mut().find(|m| m.name == map_name) {
            Some(m) => m,
            None => return false,
        };
        for (action_name, bindings) in overrides {
            if let Some(act) = map.find_action_mut(action_name) {
                act.bindings = bindings.clone();
            }
        }
        true
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::action::ActionDef;
    use crate::input::action_map::ActionMap;
    use crate::input::binding::Binding;

    fn gameplay_map() -> ActionMap {
        ActionMap::new("gameplay", vec![
            ActionDef::new("jump", ControlType::Digital, vec![
                Binding::simple(PhysicalInput::Key(InputKeyCode::Space)),
            ]),
            ActionDef::new("move", ControlType::Axis2D, vec![
                Binding::Composite2D {
                    up: PhysicalInput::Key(InputKeyCode::W),
                    down: PhysicalInput::Key(InputKeyCode::S),
                    left: PhysicalInput::Key(InputKeyCode::A),
                    right: PhysicalInput::Key(InputKeyCode::D),
                },
            ]),
            ActionDef::new("strafe", ControlType::Axis1D, vec![
                Binding::CompositeAxis {
                    positive: PhysicalInput::Key(InputKeyCode::D),
                    negative: PhysicalInput::Key(InputKeyCode::A),
                },
            ]),
        ])
    }

    #[test]
    fn end_to_end_digital_action() {
        let mut sys = InputSystem::new();
        sys.add_map(gameplay_map());
        assert!(sys.set_active_map("gameplay"));

        // Frame 1: press space
        sys.begin_frame();
        sys.feed_key_down(InputKeyCode::Space);
        sys.evaluate();

        assert!(sys.pressed("jump"));
        assert!(sys.just_pressed("jump"));
        assert_eq!(sys.action("jump").unwrap().phase, ActionPhase::Started);

        // Frame 2: still held → Performed
        sys.begin_frame();
        sys.evaluate();

        assert!(sys.pressed("jump"));
        assert!(!sys.just_pressed("jump"));
        assert_eq!(sys.action("jump").unwrap().phase, ActionPhase::Performed);
        assert!(sys.action("jump").unwrap().performed_this_frame);

        // Frame 3: still held → Performed (no performed_this_frame)
        sys.begin_frame();
        sys.evaluate();
        assert_eq!(sys.action("jump").unwrap().phase, ActionPhase::Performed);
        assert!(!sys.action("jump").unwrap().performed_this_frame);

        // Frame 4: release
        sys.begin_frame();
        sys.feed_key_up(InputKeyCode::Space);
        sys.evaluate();

        assert!(!sys.pressed("jump"));
        assert!(sys.just_released("jump"));
        assert_eq!(sys.action("jump").unwrap().phase, ActionPhase::Canceled);

        // Frame 5: nothing → Waiting
        sys.begin_frame();
        sys.evaluate();
        assert_eq!(sys.action("jump").unwrap().phase, ActionPhase::Waiting);
        assert!(!sys.just_released("jump"));
    }

    #[test]
    fn composite_2d_wasd() {
        let mut sys = InputSystem::new();
        sys.add_map(gameplay_map());
        sys.set_active_map("gameplay");

        sys.begin_frame();
        sys.feed_key_down(InputKeyCode::W);
        sys.feed_key_down(InputKeyCode::D);
        sys.evaluate();

        let move_val = sys.axis_2d("move");
        assert!(move_val.x > 0.0, "should have positive X");
        assert!(move_val.y > 0.0, "should have positive Y");
        assert!((move_val.length() - 1.0).abs() < 0.01, "should be normalized");
    }

    #[test]
    fn map_switching_clears_states() {
        let mut sys = InputSystem::new();
        sys.add_map(gameplay_map());
        sys.add_map(ActionMap::new("menu", vec![
            ActionDef::new("select", ControlType::Digital, vec![
                Binding::simple(PhysicalInput::Key(InputKeyCode::Enter)),
            ]),
        ]));

        sys.set_active_map("gameplay");
        sys.begin_frame();
        sys.feed_key_down(InputKeyCode::Space);
        sys.evaluate();
        assert!(sys.pressed("jump"));

        // Switch map — action states clear
        sys.set_active_map("menu");
        assert!(sys.action("jump").is_none());
        assert!(sys.action("select").is_none()); // not evaluated yet
    }

    #[test]
    fn inactive_action_returns_none() {
        let sys = InputSystem::new();
        assert!(sys.action("nonexistent").is_none());
        assert!(!sys.pressed("nonexistent"));
        assert!(!sys.just_pressed("nonexistent"));
        assert_eq!(sys.axis_1d("nonexistent"), 0.0);
        assert_eq!(sys.axis_2d("nonexistent"), Vec2::ZERO);
    }

    #[test]
    fn set_active_map_not_found() {
        let mut sys = InputSystem::new();
        assert!(!sys.set_active_map("nope"));
        assert!(sys.active_map().is_none());
    }

    #[test]
    fn multiple_bindings_or_for_digital() {
        let mut sys = InputSystem::new();
        sys.add_map(ActionMap::new("test", vec![
            ActionDef::new("fire", ControlType::Digital, vec![
                Binding::simple(PhysicalInput::Key(InputKeyCode::Space)),
                Binding::simple(PhysicalInput::MouseButton(InputMouseButton::Left)),
            ]),
        ]));
        sys.set_active_map("test");

        // Only mouse button
        sys.begin_frame();
        sys.feed_mouse_button(InputMouseButton::Left, true);
        sys.evaluate();
        assert!(sys.pressed("fire"));

        // Release mouse, press space
        sys.begin_frame();
        sys.feed_mouse_button(InputMouseButton::Left, false);
        sys.feed_key_down(InputKeyCode::Space);
        sys.evaluate();
        assert!(sys.pressed("fire"));
    }

    #[test]
    fn dead_zone_filtering() {
        let mut sys = InputSystem::with_dead_zone(0.2);
        sys.add_map(ActionMap::new("test", vec![
            ActionDef::new(
                "look",
                ControlType::Axis2D,
                vec![Binding::simple(PhysicalInput::GamepadStick(GamepadStick::Right))],
            ).with_dead_zone(0.15),
        ]));
        sys.set_active_map("test");

        // Below dead zone
        sys.begin_frame();
        sys.feed_gamepad_stick(GamepadStick::Right, Vec2::new(0.1, 0.05));
        sys.evaluate();
        assert_eq!(sys.axis_2d("look"), Vec2::ZERO);

        // Above dead zone
        sys.begin_frame();
        sys.feed_gamepad_stick(GamepadStick::Right, Vec2::new(0.5, 0.3));
        sys.evaluate();
        assert_ne!(sys.axis_2d("look"), Vec2::ZERO);
    }

    #[test]
    fn gamepad_button_digital_action() {
        let mut sys = InputSystem::new();
        sys.add_map(ActionMap::new("gamepad", vec![
            ActionDef::new("confirm", ControlType::Digital, vec![
                Binding::simple(PhysicalInput::GamepadButton(GamepadButton::South)),
            ]),
        ]));
        sys.set_active_map("gamepad");

        sys.begin_frame();
        sys.feed_gamepad_button(GamepadButton::South, true);
        sys.evaluate();
        assert!(sys.pressed("confirm"));
        assert!(sys.just_pressed("confirm"));

        sys.begin_frame();
        sys.feed_gamepad_button(GamepadButton::South, false);
        sys.evaluate();
        assert!(!sys.pressed("confirm"));
        assert!(sys.just_released("confirm"));
    }

    #[test]
    fn gamepad_stick_axis2d_action() {
        let mut sys = InputSystem::new();
        sys.add_map(ActionMap::new("test", vec![
            ActionDef::new(
                "move",
                ControlType::Axis2D,
                vec![Binding::simple(PhysicalInput::GamepadStick(GamepadStick::Left))],
            ).with_dead_zone(0.05),
        ]));
        sys.set_active_map("test");

        sys.begin_frame();
        sys.feed_gamepad_stick(GamepadStick::Left, Vec2::new(0.8, -0.6));
        sys.evaluate();

        let val = sys.axis_2d("move");
        assert_eq!(val, Vec2::new(0.8, -0.6));
    }

    #[test]
    fn phase_lifecycle_full_cycle() {
        let mut sys = InputSystem::new();
        sys.add_map(ActionMap::new("test", vec![
            ActionDef::new("act", ControlType::Digital, vec![
                Binding::simple(PhysicalInput::Key(InputKeyCode::X)),
            ]),
        ]));
        sys.set_active_map("test");

        // Waiting initially (after first evaluate)
        sys.begin_frame();
        sys.evaluate();
        assert_eq!(sys.action("act").unwrap().phase, ActionPhase::Waiting);

        // Press → Started
        sys.begin_frame();
        sys.feed_key_down(InputKeyCode::X);
        sys.evaluate();
        assert_eq!(sys.action("act").unwrap().phase, ActionPhase::Started);

        // Hold → Performed
        sys.begin_frame();
        sys.evaluate();
        assert_eq!(sys.action("act").unwrap().phase, ActionPhase::Performed);

        // Release → Canceled
        sys.begin_frame();
        sys.feed_key_up(InputKeyCode::X);
        sys.evaluate();
        assert_eq!(sys.action("act").unwrap().phase, ActionPhase::Canceled);

        // Nothing → Waiting
        sys.begin_frame();
        sys.evaluate();
        assert_eq!(sys.action("act").unwrap().phase, ActionPhase::Waiting);
    }

    #[test]
    fn raw_state_accessible() {
        let mut sys = InputSystem::new();
        sys.feed_key_down(InputKeyCode::W);
        assert!(sys.raw_state().is_key_pressed(InputKeyCode::W));
    }

    #[test]
    fn no_active_map_evaluate_noop() {
        let mut sys = InputSystem::new();
        sys.add_map(gameplay_map());
        // Don't set active map
        sys.begin_frame();
        sys.feed_key_down(InputKeyCode::Space);
        sys.evaluate();
        assert!(sys.action("jump").is_none());
    }

    #[test]
    fn composite_axis_1d() {
        let mut sys = InputSystem::new();
        sys.add_map(gameplay_map());
        sys.set_active_map("gameplay");

        sys.begin_frame();
        sys.feed_key_down(InputKeyCode::D);
        sys.evaluate();
        assert_eq!(sys.axis_1d("strafe"), 1.0);

        sys.begin_frame();
        sys.feed_key_down(InputKeyCode::A);
        sys.evaluate();
        // Both pressed → cancel
        assert_eq!(sys.axis_1d("strafe"), 0.0);
    }

    #[test]
    fn tap_release_same_frame_still_starts() {
        // If key is pressed and released in the same frame before evaluate,
        // the key won't be in keys_pressed (key_up removes it), so action won't fire.
        // This is expected behavior — evaluate sees instantaneous state.
        let mut sys = InputSystem::new();
        sys.add_map(ActionMap::new("test", vec![
            ActionDef::new("tap", ControlType::Digital, vec![
                Binding::simple(PhysicalInput::Key(InputKeyCode::T)),
            ]),
        ]));
        sys.set_active_map("test");

        sys.begin_frame();
        sys.feed_key_down(InputKeyCode::T);
        sys.feed_key_up(InputKeyCode::T);
        sys.evaluate();
        // Key is not pressed at evaluate time
        assert!(!sys.pressed("tap"));
    }

    // --- Rebinding tests ---

    #[test]
    fn set_binding_replaces_correctly() {
        let mut sys = InputSystem::new();
        sys.add_map(gameplay_map());
        sys.set_active_map("gameplay");

        assert!(sys.set_binding(
            "jump",
            0,
            Binding::simple(PhysicalInput::Key(InputKeyCode::Enter)),
        ));
        let bindings = sys.get_bindings("jump").unwrap();
        assert_eq!(bindings[0], Binding::simple(PhysicalInput::Key(InputKeyCode::Enter)));

        // Out of bounds returns false
        assert!(!sys.set_binding("jump", 5, Binding::simple(PhysicalInput::Key(InputKeyCode::A))));
        // Nonexistent action returns false
        assert!(!sys.set_binding("nope", 0, Binding::simple(PhysicalInput::Key(InputKeyCode::A))));
    }

    #[test]
    fn set_binding_part_changes_composite_leg() {
        let mut sys = InputSystem::new();
        sys.add_map(gameplay_map());
        sys.set_active_map("gameplay");

        assert!(sys.set_binding_part("move", 0, "up", PhysicalInput::Key(InputKeyCode::ArrowUp)));

        let bindings = sys.get_bindings("move").unwrap();
        match &bindings[0] {
            Binding::Composite2D { up, .. } => {
                assert_eq!(*up, PhysicalInput::Key(InputKeyCode::ArrowUp));
            }
            _ => panic!("expected Composite2D"),
        }

        // Invalid part name returns false
        assert!(!sys.set_binding_part("move", 0, "invalid", PhysicalInput::Key(InputKeyCode::X)));

        // Simple binding doesn't have parts
        assert!(!sys.set_binding_part("jump", 0, "up", PhysicalInput::Key(InputKeyCode::X)));
    }

    #[test]
    fn set_binding_part_composite_axis() {
        let mut sys = InputSystem::new();
        sys.add_map(gameplay_map());
        sys.set_active_map("gameplay");

        assert!(sys.set_binding_part(
            "strafe",
            0,
            "positive",
            PhysicalInput::Key(InputKeyCode::ArrowRight),
        ));
        let bindings = sys.get_bindings("strafe").unwrap();
        match &bindings[0] {
            Binding::CompositeAxis { positive, negative } => {
                assert_eq!(*positive, PhysicalInput::Key(InputKeyCode::ArrowRight));
                assert_eq!(*negative, PhysicalInput::Key(InputKeyCode::A));
            }
            _ => panic!("expected CompositeAxis"),
        }
    }

    #[test]
    fn add_binding_appends() {
        let mut sys = InputSystem::new();
        sys.add_map(gameplay_map());
        sys.set_active_map("gameplay");

        assert_eq!(sys.get_bindings("jump").unwrap().len(), 1);
        assert!(sys.add_binding("jump", Binding::simple(PhysicalInput::Key(InputKeyCode::Enter))));
        assert_eq!(sys.get_bindings("jump").unwrap().len(), 2);
        assert!(!sys.add_binding("nope", Binding::simple(PhysicalInput::Key(InputKeyCode::A))));
    }

    #[test]
    fn remove_binding_removes() {
        let mut sys = InputSystem::new();
        sys.add_map(gameplay_map());
        sys.set_active_map("gameplay");

        sys.add_binding("jump", Binding::simple(PhysicalInput::Key(InputKeyCode::Enter)));
        assert_eq!(sys.get_bindings("jump").unwrap().len(), 2);

        assert!(sys.remove_binding("jump", 0));
        assert_eq!(sys.get_bindings("jump").unwrap().len(), 1);
        assert_eq!(
            sys.get_bindings("jump").unwrap()[0],
            Binding::simple(PhysicalInput::Key(InputKeyCode::Enter)),
        );

        // Out of bounds
        assert!(!sys.remove_binding("jump", 5));
        assert!(!sys.remove_binding("nope", 0));
    }

    #[test]
    fn find_conflicts_detects_shared_keys() {
        let mut sys = InputSystem::new();
        sys.add_map(gameplay_map());
        sys.set_active_map("gameplay");

        // Space is used by "jump"
        let conflicts = sys.find_conflicts(&Binding::simple(PhysicalInput::Key(InputKeyCode::Space)));
        assert!(conflicts.contains(&"jump".to_string()));
        assert!(!conflicts.contains(&"move".to_string()));

        // D is used by "move" (composite right) and "strafe" (composite positive)
        let conflicts = sys.find_conflicts(&Binding::simple(PhysicalInput::Key(InputKeyCode::D)));
        assert!(conflicts.contains(&"move".to_string()));
        assert!(conflicts.contains(&"strafe".to_string()));

        // No conflicts for unused key
        let conflicts = sys.find_conflicts(&Binding::simple(PhysicalInput::Key(InputKeyCode::Z)));
        assert!(conflicts.is_empty());
    }

    #[test]
    fn reset_binding_restores_defaults() {
        let mut sys = InputSystem::new();
        sys.add_map(gameplay_map());
        sys.set_active_map("gameplay");

        let original = sys.get_bindings("jump").unwrap()[0].clone();
        sys.set_binding("jump", 0, Binding::simple(PhysicalInput::Key(InputKeyCode::Enter)));
        assert_ne!(sys.get_bindings("jump").unwrap()[0], original);

        assert!(sys.reset_binding("jump"));
        assert_eq!(sys.get_bindings("jump").unwrap()[0], original);

        assert!(!sys.reset_binding("nope"));
    }

    #[test]
    fn reset_active_map_restores_all() {
        let mut sys = InputSystem::new();
        sys.add_map(gameplay_map());
        sys.set_active_map("gameplay");

        sys.set_binding("jump", 0, Binding::simple(PhysicalInput::Key(InputKeyCode::Enter)));
        sys.add_binding("move", Binding::simple(PhysicalInput::Key(InputKeyCode::ArrowUp)));

        assert!(sys.reset_active_map());
        assert_eq!(sys.get_bindings("jump").unwrap().len(), 1);
        assert_eq!(
            sys.get_bindings("jump").unwrap()[0],
            Binding::simple(PhysicalInput::Key(InputKeyCode::Space)),
        );
        assert_eq!(sys.get_bindings("move").unwrap().len(), 1);
    }

    #[test]
    fn export_overrides_only_includes_changed() {
        let mut sys = InputSystem::new();
        sys.add_map(gameplay_map());
        sys.set_active_map("gameplay");

        // No changes -> empty overrides
        let overrides = sys.export_overrides("gameplay").unwrap();
        assert!(overrides.is_empty());

        // Change jump binding
        sys.set_binding("jump", 0, Binding::simple(PhysicalInput::Key(InputKeyCode::Enter)));
        let overrides = sys.export_overrides("gameplay").unwrap();
        assert_eq!(overrides.len(), 1);
        assert!(overrides.contains_key("jump"));
        assert!(!overrides.contains_key("move"));

        // Nonexistent map
        assert!(sys.export_overrides("nope").is_none());
    }

    #[test]
    fn apply_overrides_modifies_bindings() {
        let mut sys = InputSystem::new();
        sys.add_map(gameplay_map());
        sys.set_active_map("gameplay");

        let mut overrides = HashMap::new();
        overrides.insert(
            "jump".to_string(),
            vec![
                Binding::simple(PhysicalInput::Key(InputKeyCode::Enter)),
                Binding::simple(PhysicalInput::GamepadButton(GamepadButton::South)),
            ],
        );

        assert!(sys.apply_overrides("gameplay", &overrides));
        assert_eq!(sys.get_bindings("jump").unwrap().len(), 2);
        assert_eq!(
            sys.get_bindings("jump").unwrap()[0],
            Binding::simple(PhysicalInput::Key(InputKeyCode::Enter)),
        );

        assert!(!sys.apply_overrides("nope", &overrides));
    }

    #[test]
    fn rebind_then_evaluate_uses_new_binding() {
        let mut sys = InputSystem::new();
        sys.add_map(gameplay_map());
        sys.set_active_map("gameplay");

        // Rebind jump from Space to Enter
        sys.set_binding("jump", 0, Binding::simple(PhysicalInput::Key(InputKeyCode::Enter)));

        // Space should no longer trigger jump
        sys.begin_frame();
        sys.feed_key_down(InputKeyCode::Space);
        sys.evaluate();
        assert!(!sys.pressed("jump"));

        // Enter should trigger jump
        sys.begin_frame();
        sys.feed_key_down(InputKeyCode::Enter);
        sys.evaluate();
        assert!(sys.pressed("jump"));
    }

    #[test]
    fn get_bindings_no_active_map() {
        let sys = InputSystem::new();
        assert!(sys.get_bindings("jump").is_none());
    }

    #[test]
    fn find_conflicts_no_active_map() {
        let sys = InputSystem::new();
        let conflicts = sys.find_conflicts(&Binding::simple(PhysicalInput::Key(InputKeyCode::A)));
        assert!(conflicts.is_empty());
    }

    #[test]
    fn reset_active_map_no_active() {
        let mut sys = InputSystem::new();
        assert!(!sys.reset_active_map());
    }

    #[test]
    fn enable_gamepad_succeeds() {
        let mut sys = InputSystem::new();
        // May return false on CI without gamepad subsystem, but should not panic
        let _ = sys.enable_gamepad();
        // Calling again is idempotent
        if sys.enable_gamepad() {
            assert!(sys.enable_gamepad());
        }
    }

    #[test]
    fn gamepad_ui_events_from_buttons() {
        let mut sys = InputSystem::new();
        sys.feed_gamepad_button(GamepadButton::DPadUp, true);
        sys.feed_gamepad_button(GamepadButton::South, true);
        let events = sys.gamepad_ui_events();
        assert_eq!(events.len(), 2);
        let keys: std::collections::HashSet<InputKeyCode> =
            events.iter().map(|e| e.key).collect();
        assert!(keys.contains(&InputKeyCode::ArrowUp));
        assert!(keys.contains(&InputKeyCode::Enter));
    }

    #[test]
    fn gamepad_ui_events_released() {
        let mut sys = InputSystem::new();
        // Press on frame 1
        sys.feed_gamepad_button(GamepadButton::East, true);
        sys.begin_frame();
        // Release on frame 2
        sys.feed_gamepad_button(GamepadButton::East, false);
        let events = sys.gamepad_ui_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].key, InputKeyCode::Escape);
        assert!(!events[0].pressed);
    }

    #[test]
    fn connected_gamepads_empty_without_enable() {
        let sys = InputSystem::new();
        assert!(sys.connected_gamepads().is_empty());
    }

    #[test]
    fn poll_gamepads_noop_without_enable() {
        let mut sys = InputSystem::new();
        sys.poll_gamepads(); // should not panic
    }

    #[test]
    fn shortcut_strings_returns_first_binding_display() {
        let sys = {
            let mut s = InputSystem::new();
            s.add_map(ActionMap::new("test", vec![
                ActionDef::new("do.jump", ControlType::Digital, vec![
                    Binding::simple(PhysicalInput::Key(InputKeyCode::Space)),
                ]),
                ActionDef::new("do.save", ControlType::Digital, vec![
                    Binding::simple_with_mod(
                        PhysicalInput::Key(InputKeyCode::S),
                        ModifierMask::ctrl(),
                    ),
                ]),
                ActionDef::new("do.empty", ControlType::Digital, vec![]),
            ]));
            s
        };

        let shortcuts = sys.shortcut_strings("test");
        assert_eq!(shortcuts.get("do.jump").map(|s| s.as_str()), Some("Space"));
        assert_eq!(shortcuts.get("do.save").map(|s| s.as_str()), Some("Ctrl+S"));
        assert!(!shortcuts.contains_key("do.empty"), "actions with no bindings should be omitted");
    }

    #[test]
    fn shortcut_strings_nonexistent_map() {
        let sys = InputSystem::new();
        let shortcuts = sys.shortcut_strings("nonexistent");
        assert!(shortcuts.is_empty());
    }
}
