//! Core input system types — control types, action phases, key codes, physical inputs.

use glam::Vec2;
use serde::{Serialize, Deserialize};

/// What kind of value an action produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ControlType {
    /// On/off — button press.
    Digital,
    /// Single-axis float — trigger, scroll.
    Axis1D,
    /// Two-axis float — stick, WASD composite.
    Axis2D,
}

/// Lifecycle phase of an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionPhase {
    /// Not active.
    Waiting,
    /// Just became active this frame.
    Started,
    /// Held beyond the start frame.
    Performed,
    /// Just released this frame.
    Canceled,
}

/// Current state of a single action.
#[derive(Debug, Clone)]
pub struct ActionState {
    /// Current lifecycle phase.
    pub phase: ActionPhase,
    /// Digital (on/off) value.
    pub digital: bool,
    /// Single-axis value.
    pub axis_1d: f32,
    /// Two-axis value.
    pub axis_2d: Vec2,
    /// True only on the frame the action entered Started.
    pub started_this_frame: bool,
    /// True only on the frame the action entered Performed.
    pub performed_this_frame: bool,
    /// True only on the frame the action entered Canceled.
    pub canceled_this_frame: bool,
}

impl Default for ActionState {
    fn default() -> Self {
        Self {
            phase: ActionPhase::Waiting,
            digital: false,
            axis_1d: 0.0,
            axis_2d: Vec2::ZERO,
            started_this_frame: false,
            performed_this_frame: false,
            canceled_this_frame: false,
        }
    }
}

/// Keyboard key codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum InputKeyCode {
    A, B, C, D, E, F, G, H, I, J, K, L, M,
    N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
    Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,
    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
    ArrowUp, ArrowDown, ArrowLeft, ArrowRight,
    Space, Tab, Enter, Escape,
    Delete, Backspace,
    ShiftLeft, ShiftRight,
    ControlLeft, ControlRight,
    AltLeft, AltRight,
    Comma, Period, Slash, Semicolon, Quote,
    BracketLeft, BracketRight, Backslash,
    Minus, Equal,
    Home, End, PageUp, PageDown, Insert,
    CapsLock,
    NumpadAdd, NumpadSubtract, NumpadMultiply, NumpadDivide, NumpadEnter,
    Numpad0, Numpad1, Numpad2, Numpad3, Numpad4,
    Numpad5, Numpad6, Numpad7, Numpad8, Numpad9,
    Grave,
}

/// Mouse buttons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InputMouseButton {
    /// Left mouse button.
    Left,
    /// Right mouse button.
    Right,
    /// Middle mouse button.
    Middle,
}

/// Gamepad face and shoulder buttons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum GamepadButton {
    South, East, West, North,
    DPadUp, DPadDown, DPadLeft, DPadRight,
    LeftBumper, RightBumper,
    LeftStick, RightStick,
    Start, Select, Guide,
}

/// Gamepad analog axes (triggers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum GamepadAxis {
    LeftTrigger,
    RightTrigger,
}

/// Gamepad analog sticks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum GamepadStick {
    Left,
    Right,
}

/// Modifier key requirements for a binding.
///
/// Each field is tri-state: `None` = don't care, `Some(true)` = must be held,
/// `Some(false)` = must NOT be held.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct ModifierMask {
    /// Control key requirement.
    pub ctrl: Option<bool>,
    /// Shift key requirement.
    pub shift: Option<bool>,
    /// Alt key requirement.
    pub alt: Option<bool>,
}

impl ModifierMask {
    /// No modifier requirements (matches any modifier state).
    pub fn none() -> Self {
        Self::default()
    }

    /// Require Ctrl held.
    pub fn ctrl() -> Self {
        Self { ctrl: Some(true), ..Self::default() }
    }

    /// Require Shift held.
    pub fn shift() -> Self {
        Self { shift: Some(true), ..Self::default() }
    }

    /// Require Alt held.
    pub fn alt() -> Self {
        Self { alt: Some(true), ..Self::default() }
    }

    /// Require Ctrl+Shift held.
    pub fn ctrl_shift() -> Self {
        Self { ctrl: Some(true), shift: Some(true), ..Self::default() }
    }

    /// Check if current modifier state matches this mask.
    pub fn matches(&self, ctrl_held: bool, shift_held: bool, alt_held: bool) -> bool {
        self.ctrl.is_none_or(|req| req == ctrl_held)
            && self.shift.is_none_or(|req| req == shift_held)
            && self.alt.is_none_or(|req| req == alt_held)
    }

    /// Specificity score (more specific masks win conflicts).
    pub fn specificity(&self) -> u8 {
        self.ctrl.is_some() as u8 + self.shift.is_some() as u8 + self.alt.is_some() as u8
    }
}

/// A physical input source — keys, mouse, gamepad.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PhysicalInput {
    /// Keyboard key.
    Key(InputKeyCode),
    /// Mouse button.
    MouseButton(InputMouseButton),
    /// Mouse movement delta.
    MouseDelta,
    /// Scroll wheel.
    ScrollWheel,
    /// Gamepad button.
    GamepadButton(GamepadButton),
    /// Gamepad analog axis (trigger).
    GamepadAxis(GamepadAxis),
    /// Gamepad analog stick.
    GamepadStick(GamepadStick),
}

impl PhysicalInput {
    /// Human-readable name for this input (e.g. "W", "LMB", "Left Stick").
    pub fn display_name(&self) -> &'static str {
        match self {
            PhysicalInput::Key(k) => k.display_name(),
            PhysicalInput::MouseButton(InputMouseButton::Left) => "LMB",
            PhysicalInput::MouseButton(InputMouseButton::Right) => "RMB",
            PhysicalInput::MouseButton(InputMouseButton::Middle) => "MMB",
            PhysicalInput::MouseDelta => "Mouse",
            PhysicalInput::ScrollWheel => "Scroll",
            PhysicalInput::GamepadButton(b) => match b {
                GamepadButton::South => "A",
                GamepadButton::East => "B",
                GamepadButton::West => "X",
                GamepadButton::North => "Y",
                GamepadButton::DPadUp => "DPad Up",
                GamepadButton::DPadDown => "DPad Down",
                GamepadButton::DPadLeft => "DPad Left",
                GamepadButton::DPadRight => "DPad Right",
                GamepadButton::LeftBumper => "LB",
                GamepadButton::RightBumper => "RB",
                GamepadButton::LeftStick => "LS",
                GamepadButton::RightStick => "RS",
                GamepadButton::Start => "Start",
                GamepadButton::Select => "Select",
                GamepadButton::Guide => "Guide",
            },
            PhysicalInput::GamepadAxis(GamepadAxis::LeftTrigger) => "LT",
            PhysicalInput::GamepadAxis(GamepadAxis::RightTrigger) => "RT",
            PhysicalInput::GamepadStick(GamepadStick::Left) => "Left Stick",
            PhysicalInput::GamepadStick(GamepadStick::Right) => "Right Stick",
        }
    }
}

impl InputKeyCode {
    /// Human-readable name for display in UI.
    pub fn display_name(&self) -> &'static str {
        match self {
            InputKeyCode::A => "A", InputKeyCode::B => "B", InputKeyCode::C => "C",
            InputKeyCode::D => "D", InputKeyCode::E => "E", InputKeyCode::F => "F",
            InputKeyCode::G => "G", InputKeyCode::H => "H", InputKeyCode::I => "I",
            InputKeyCode::J => "J", InputKeyCode::K => "K", InputKeyCode::L => "L",
            InputKeyCode::M => "M", InputKeyCode::N => "N", InputKeyCode::O => "O",
            InputKeyCode::P => "P", InputKeyCode::Q => "Q", InputKeyCode::R => "R",
            InputKeyCode::S => "S", InputKeyCode::T => "T", InputKeyCode::U => "U",
            InputKeyCode::V => "V", InputKeyCode::W => "W", InputKeyCode::X => "X",
            InputKeyCode::Y => "Y", InputKeyCode::Z => "Z",
            InputKeyCode::Num0 => "0", InputKeyCode::Num1 => "1", InputKeyCode::Num2 => "2",
            InputKeyCode::Num3 => "3", InputKeyCode::Num4 => "4", InputKeyCode::Num5 => "5",
            InputKeyCode::Num6 => "6", InputKeyCode::Num7 => "7", InputKeyCode::Num8 => "8",
            InputKeyCode::Num9 => "9",
            InputKeyCode::F1 => "F1", InputKeyCode::F2 => "F2", InputKeyCode::F3 => "F3",
            InputKeyCode::F4 => "F4", InputKeyCode::F5 => "F5", InputKeyCode::F6 => "F6",
            InputKeyCode::F7 => "F7", InputKeyCode::F8 => "F8", InputKeyCode::F9 => "F9",
            InputKeyCode::F10 => "F10", InputKeyCode::F11 => "F11", InputKeyCode::F12 => "F12",
            InputKeyCode::ArrowUp => "Up", InputKeyCode::ArrowDown => "Down",
            InputKeyCode::ArrowLeft => "Left", InputKeyCode::ArrowRight => "Right",
            InputKeyCode::Space => "Space", InputKeyCode::Tab => "Tab",
            InputKeyCode::Enter => "Enter", InputKeyCode::Escape => "Esc",
            InputKeyCode::Delete => "Del", InputKeyCode::Backspace => "Backspace",
            InputKeyCode::ShiftLeft | InputKeyCode::ShiftRight => "Shift",
            InputKeyCode::ControlLeft | InputKeyCode::ControlRight => "Ctrl",
            InputKeyCode::AltLeft | InputKeyCode::AltRight => "Alt",
            InputKeyCode::Comma => ",", InputKeyCode::Period => ".",
            InputKeyCode::Slash => "/", InputKeyCode::Semicolon => ";",
            InputKeyCode::Quote => "'", InputKeyCode::BracketLeft => "[",
            InputKeyCode::BracketRight => "]", InputKeyCode::Backslash => "\\",
            InputKeyCode::Minus => "-", InputKeyCode::Equal => "=",
            InputKeyCode::Home => "Home", InputKeyCode::End => "End",
            InputKeyCode::PageUp => "PgUp", InputKeyCode::PageDown => "PgDn",
            InputKeyCode::Insert => "Ins", InputKeyCode::CapsLock => "CapsLock",
            InputKeyCode::NumpadAdd => "Num+", InputKeyCode::NumpadSubtract => "Num-",
            InputKeyCode::NumpadMultiply => "Num*", InputKeyCode::NumpadDivide => "Num/",
            InputKeyCode::NumpadEnter => "NumEnter",
            InputKeyCode::Numpad0 => "Num0", InputKeyCode::Numpad1 => "Num1",
            InputKeyCode::Numpad2 => "Num2", InputKeyCode::Numpad3 => "Num3",
            InputKeyCode::Numpad4 => "Num4", InputKeyCode::Numpad5 => "Num5",
            InputKeyCode::Numpad6 => "Num6", InputKeyCode::Numpad7 => "Num7",
            InputKeyCode::Numpad8 => "Num8", InputKeyCode::Numpad9 => "Num9",
            InputKeyCode::Grave => "`",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn default_action_state() {
        let state = ActionState::default();
        assert_eq!(state.phase, ActionPhase::Waiting);
        assert!(!state.digital);
        assert_eq!(state.axis_1d, 0.0);
        assert_eq!(state.axis_2d, Vec2::ZERO);
        assert!(!state.started_this_frame);
        assert!(!state.performed_this_frame);
        assert!(!state.canceled_this_frame);
    }

    #[test]
    fn control_type_equality() {
        assert_eq!(ControlType::Digital, ControlType::Digital);
        assert_ne!(ControlType::Digital, ControlType::Axis1D);
        assert_ne!(ControlType::Axis1D, ControlType::Axis2D);
    }

    #[test]
    fn physical_input_hash_key() {
        let mut set = HashSet::new();
        set.insert(PhysicalInput::Key(InputKeyCode::W));
        set.insert(PhysicalInput::Key(InputKeyCode::A));
        assert!(set.contains(&PhysicalInput::Key(InputKeyCode::W)));
        assert!(!set.contains(&PhysicalInput::Key(InputKeyCode::S)));
    }

    #[test]
    fn physical_input_variants() {
        let _k = PhysicalInput::Key(InputKeyCode::Space);
        let _m = PhysicalInput::MouseButton(InputMouseButton::Left);
        let _d = PhysicalInput::MouseDelta;
        let _s = PhysicalInput::ScrollWheel;
        let _gb = PhysicalInput::GamepadButton(GamepadButton::South);
        let _ga = PhysicalInput::GamepadAxis(GamepadAxis::LeftTrigger);
        let _gs = PhysicalInput::GamepadStick(GamepadStick::Left);
    }

    #[test]
    fn mouse_button_index() {
        // Verify all three buttons are distinct
        let buttons = [InputMouseButton::Left, InputMouseButton::Right, InputMouseButton::Middle];
        let set: HashSet<_> = buttons.iter().collect();
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn modifier_mask_none_matches_everything() {
        let mask = ModifierMask::none();
        assert!(mask.matches(false, false, false));
        assert!(mask.matches(true, false, false));
        assert!(mask.matches(true, true, true));
    }

    #[test]
    fn modifier_mask_ctrl_requires_ctrl() {
        let mask = ModifierMask::ctrl();
        assert!(!mask.matches(false, false, false));
        assert!(mask.matches(true, false, false));
        assert!(mask.matches(true, true, false)); // shift doesn't matter
    }

    #[test]
    fn modifier_mask_ctrl_shift_requires_both() {
        let mask = ModifierMask::ctrl_shift();
        assert!(!mask.matches(true, false, false));
        assert!(!mask.matches(false, true, false));
        assert!(mask.matches(true, true, false));
        assert!(mask.matches(true, true, true)); // alt doesn't matter
    }

    #[test]
    fn modifier_mask_specificity() {
        assert_eq!(ModifierMask::none().specificity(), 0);
        assert_eq!(ModifierMask::ctrl().specificity(), 1);
        assert_eq!(ModifierMask::ctrl_shift().specificity(), 2);
        let full = ModifierMask { ctrl: Some(true), shift: Some(true), alt: Some(true) };
        assert_eq!(full.specificity(), 3);
    }
}
