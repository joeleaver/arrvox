//! Input bindings — maps physical inputs to action values.

use glam::Vec2;
use serde::{Serialize, Deserialize};

use super::raw_state::RawInputState;
use super::types::{InputKeyCode, ModifierMask, PhysicalInput};

/// A binding that maps one or more physical inputs to an action value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Binding {
    /// A single physical input mapped directly, with optional modifier requirements.
    Simple {
        /// The physical input source.
        input: PhysicalInput,
        /// Modifier key requirements.
        modifiers: ModifierMask,
    },
    /// Four digital inputs composited into a 2D vector (e.g., WASD).
    Composite2D {
        /// Input for +Y.
        up: PhysicalInput,
        /// Input for -Y.
        down: PhysicalInput,
        /// Input for -X.
        left: PhysicalInput,
        /// Input for +X.
        right: PhysicalInput,
    },
    /// Two digital inputs composited into a 1D axis.
    CompositeAxis {
        /// Input for positive direction.
        positive: PhysicalInput,
        /// Input for negative direction.
        negative: PhysicalInput,
    },
}

impl Binding {
    /// Create a simple binding with no modifier requirements.
    pub fn simple(input: PhysicalInput) -> Self {
        Binding::Simple { input, modifiers: ModifierMask::none() }
    }

    /// Create a simple binding with modifier requirements.
    pub fn simple_with_mod(input: PhysicalInput, modifiers: ModifierMask) -> Self {
        Binding::Simple { input, modifiers }
    }

    /// Get a human-readable display string for this binding (e.g. "Ctrl+Z", "WASD").
    /// Used for menu shortcut labels and rebinding UI.
    pub fn display_string(&self) -> String {
        match self {
            Binding::Simple { input, modifiers } => {
                let mut parts = Vec::new();
                if modifiers.ctrl == Some(true) { parts.push("Ctrl"); }
                if modifiers.shift == Some(true) { parts.push("Shift"); }
                if modifiers.alt == Some(true) { parts.push("Alt"); }
                parts.push(input.display_name());
                parts.join("+")
            }
            Binding::Composite2D { up, down, left, right } => {
                format!("{}/{}/{}/{}",
                    up.display_name(), left.display_name(),
                    down.display_name(), right.display_name())
            }
            Binding::CompositeAxis { positive, negative } => {
                format!("{}/{}", positive.display_name(), negative.display_name())
            }
        }
    }

    /// Evaluate this binding against raw input state.
    /// Returns (digital, axis_1d, axis_2d).
    pub fn evaluate(&self, raw: &RawInputState) -> (bool, f32, Vec2) {
        match self {
            Binding::Simple { input, modifiers } => {
                // Check modifier requirements.
                let ctrl_held = raw.is_key_pressed(InputKeyCode::ControlLeft)
                    || raw.is_key_pressed(InputKeyCode::ControlRight);
                let shift_held = raw.is_key_pressed(InputKeyCode::ShiftLeft)
                    || raw.is_key_pressed(InputKeyCode::ShiftRight);
                let alt_held = raw.is_key_pressed(InputKeyCode::AltLeft)
                    || raw.is_key_pressed(InputKeyCode::AltRight);

                if !modifiers.matches(ctrl_held, shift_held, alt_held) {
                    return (false, 0.0, Vec2::ZERO);
                }

                raw.physical_input_value(input)
            }
            Binding::Composite2D { up, down, left, right } => {
                let u = if raw.is_physical_input_active(up) { 1.0f32 } else { 0.0 };
                let d = if raw.is_physical_input_active(down) { 1.0f32 } else { 0.0 };
                let l = if raw.is_physical_input_active(left) { 1.0f32 } else { 0.0 };
                let r = if raw.is_physical_input_active(right) { 1.0f32 } else { 0.0 };

                let mut vec = Vec2::new(r - l, u - d);
                let mag = vec.length();
                if mag > 1.0 {
                    vec /= mag;
                }

                let digital = mag > 0.0;
                (digital, mag.min(1.0), vec)
            }
            Binding::CompositeAxis { positive, negative } => {
                let pos = if raw.is_physical_input_active(positive) { 1.0f32 } else { 0.0 };
                let neg = if raw.is_physical_input_active(negative) { 1.0f32 } else { 0.0 };
                let val = pos - neg;
                (val != 0.0, val, Vec2::ZERO)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::types::InputKeyCode;

    fn make_raw() -> RawInputState {
        RawInputState::new()
    }

    #[test]
    fn simple_key_binding() {
        let binding = Binding::simple(PhysicalInput::Key(InputKeyCode::Space));
        let mut raw = make_raw();

        let (d, a, _) = binding.evaluate(&raw);
        assert!(!d);
        assert_eq!(a, 0.0);

        raw.key_down(InputKeyCode::Space);
        let (d, a, _) = binding.evaluate(&raw);
        assert!(d);
        assert_eq!(a, 1.0);
    }

    #[test]
    fn composite_2d_single_direction() {
        let binding = Binding::Composite2D {
            up: PhysicalInput::Key(InputKeyCode::W),
            down: PhysicalInput::Key(InputKeyCode::S),
            left: PhysicalInput::Key(InputKeyCode::A),
            right: PhysicalInput::Key(InputKeyCode::D),
        };
        let mut raw = make_raw();

        // Press W only → (0, 1)
        raw.key_down(InputKeyCode::W);
        let (d, a, v) = binding.evaluate(&raw);
        assert!(d);
        assert_eq!(a, 1.0);
        assert_eq!(v, Vec2::new(0.0, 1.0));
    }

    #[test]
    fn composite_2d_diagonal_normalized() {
        let binding = Binding::Composite2D {
            up: PhysicalInput::Key(InputKeyCode::W),
            down: PhysicalInput::Key(InputKeyCode::S),
            left: PhysicalInput::Key(InputKeyCode::A),
            right: PhysicalInput::Key(InputKeyCode::D),
        };
        let mut raw = make_raw();

        // Press W + D → diagonal, should be normalized
        raw.key_down(InputKeyCode::W);
        raw.key_down(InputKeyCode::D);
        let (d, a, v) = binding.evaluate(&raw);
        assert!(d);
        // Magnitude should be clamped to 1.0
        assert!((a - 1.0).abs() < 0.001);
        // Vec should be normalized
        assert!((v.length() - 1.0).abs() < 0.001);
        assert!(v.x > 0.0);
        assert!(v.y > 0.0);
    }

    #[test]
    fn composite_2d_opposing_cancel() {
        let binding = Binding::Composite2D {
            up: PhysicalInput::Key(InputKeyCode::W),
            down: PhysicalInput::Key(InputKeyCode::S),
            left: PhysicalInput::Key(InputKeyCode::A),
            right: PhysicalInput::Key(InputKeyCode::D),
        };
        let mut raw = make_raw();

        // Press W + S → cancel out
        raw.key_down(InputKeyCode::W);
        raw.key_down(InputKeyCode::S);
        let (d, _, v) = binding.evaluate(&raw);
        assert!(!d);
        assert_eq!(v, Vec2::ZERO);
    }

    #[test]
    fn composite_axis() {
        let binding = Binding::CompositeAxis {
            positive: PhysicalInput::Key(InputKeyCode::D),
            negative: PhysicalInput::Key(InputKeyCode::A),
        };
        let mut raw = make_raw();

        // Nothing pressed
        let (d, a, _) = binding.evaluate(&raw);
        assert!(!d);
        assert_eq!(a, 0.0);

        // Positive only
        raw.key_down(InputKeyCode::D);
        let (d, a, _) = binding.evaluate(&raw);
        assert!(d);
        assert_eq!(a, 1.0);

        // Both pressed → cancel
        raw.key_down(InputKeyCode::A);
        let (d, a, _) = binding.evaluate(&raw);
        assert!(!d);
        assert_eq!(a, 0.0);
    }

    #[test]
    fn simple_mouse_button_binding() {
        let binding = Binding::simple(PhysicalInput::MouseButton(
            crate::input::types::InputMouseButton::Left,
        ));
        let mut raw = make_raw();

        raw.mouse_button_down(crate::input::types::InputMouseButton::Left);
        let (d, a, _) = binding.evaluate(&raw);
        assert!(d);
        assert_eq!(a, 1.0);
    }

    #[test]
    fn simple_gamepad_stick_binding() {
        let binding = Binding::simple(PhysicalInput::GamepadStick(
            crate::input::types::GamepadStick::Left,
        ));
        let mut raw = make_raw();

        raw.set_gamepad_stick(crate::input::types::GamepadStick::Left, Vec2::new(0.7, -0.4));
        let (d, _, v) = binding.evaluate(&raw);
        assert!(d);
        assert_eq!(v, Vec2::new(0.7, -0.4));
    }

    #[test]
    fn modifier_binding_requires_ctrl() {
        use crate::input::types::ModifierMask;
        let binding = Binding::simple_with_mod(
            PhysicalInput::Key(InputKeyCode::Z),
            ModifierMask::ctrl(),
        );
        let mut raw = make_raw();

        // Z alone — should NOT fire
        raw.key_down(InputKeyCode::Z);
        let (d, _, _) = binding.evaluate(&raw);
        assert!(!d, "Z without Ctrl should not fire");

        // Z + Ctrl — should fire
        raw.key_down(InputKeyCode::ControlLeft);
        let (d, _, _) = binding.evaluate(&raw);
        assert!(d, "Ctrl+Z should fire");
    }

    #[test]
    fn modifier_binding_ctrl_shift_vs_ctrl() {
        use crate::input::types::ModifierMask;
        let undo = Binding::simple_with_mod(
            PhysicalInput::Key(InputKeyCode::Z),
            ModifierMask::ctrl(),
        );
        let redo = Binding::simple_with_mod(
            PhysicalInput::Key(InputKeyCode::Z),
            ModifierMask::ctrl_shift(),
        );
        let mut raw = make_raw();

        raw.key_down(InputKeyCode::Z);
        raw.key_down(InputKeyCode::ControlLeft);
        raw.key_down(InputKeyCode::ShiftLeft);

        // With Ctrl+Shift held, both bindings match (ctrl mask has None for shift = don't care)
        let (d_undo, _, _) = undo.evaluate(&raw);
        let (d_redo, _, _) = redo.evaluate(&raw);
        assert!(d_undo, "Ctrl mask (don't-care shift) matches Ctrl+Shift");
        assert!(d_redo, "Ctrl+Shift mask matches Ctrl+Shift");

        // Redo is more specific — this matters for conflict resolution at the system level
        assert!(redo.specificity() > undo.specificity());
    }

    #[test]
    fn no_modifier_binding_unaffected() {
        let binding = Binding::simple(PhysicalInput::Key(InputKeyCode::G));
        let mut raw = make_raw();

        // G alone
        raw.key_down(InputKeyCode::G);
        let (d, _, _) = binding.evaluate(&raw);
        assert!(d);

        // G + Ctrl (no modifier mask = don't care)
        raw.key_down(InputKeyCode::ControlLeft);
        let (d, _, _) = binding.evaluate(&raw);
        assert!(d, "binding with ModifierMask::none() should fire regardless of modifiers");
    }
}

impl Binding {
    /// Get the specificity of this binding's modifier mask (for conflict resolution).
    pub fn specificity(&self) -> u8 {
        match self {
            Binding::Simple { modifiers, .. } => modifiers.specificity(),
            _ => 0,
        }
    }
}

#[cfg(test)]
mod display_tests {
    use super::*;
    use crate::input::types::{InputKeyCode, InputMouseButton, ModifierMask, GamepadButton};

    #[test]
    fn display_simple_key() {
        let b = Binding::simple(PhysicalInput::Key(InputKeyCode::W));
        assert_eq!(b.display_string(), "W");
    }

    #[test]
    fn display_ctrl_z() {
        let b = Binding::simple_with_mod(
            PhysicalInput::Key(InputKeyCode::Z),
            ModifierMask::ctrl(),
        );
        assert_eq!(b.display_string(), "Ctrl+Z");
    }

    #[test]
    fn display_ctrl_shift_z() {
        let b = Binding::simple_with_mod(
            PhysicalInput::Key(InputKeyCode::Z),
            ModifierMask::ctrl_shift(),
        );
        assert_eq!(b.display_string(), "Ctrl+Shift+Z");
    }

    #[test]
    fn display_mouse_button() {
        let b = Binding::simple(PhysicalInput::MouseButton(InputMouseButton::Right));
        assert_eq!(b.display_string(), "RMB");
    }

    #[test]
    fn display_gamepad_button() {
        let b = Binding::simple(PhysicalInput::GamepadButton(GamepadButton::South));
        assert_eq!(b.display_string(), "A");
    }

    #[test]
    fn display_composite_2d() {
        let b = Binding::Composite2D {
            up: PhysicalInput::Key(InputKeyCode::W),
            down: PhysicalInput::Key(InputKeyCode::S),
            left: PhysicalInput::Key(InputKeyCode::A),
            right: PhysicalInput::Key(InputKeyCode::D),
        };
        assert_eq!(b.display_string(), "W/A/S/D");
    }
}
