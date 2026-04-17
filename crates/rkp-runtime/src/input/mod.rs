//! Input system — action maps, bindings, composites, and gamepad support.
//!
//! The input system translates raw physical inputs (keys, mouse, gamepad) into
//! named actions with digital, 1D axis, and 2D axis values. Actions are grouped
//! into action maps that can be switched at runtime (e.g., "gameplay" vs "menu").

pub mod types;
pub mod raw_state;
pub mod binding;
pub mod action;
pub mod action_map;
pub mod system;
pub mod serialize;
pub mod gamepad;
pub mod gamepad_ui;

pub use types::{
    ActionPhase, ActionState, ControlType, GamepadAxis, GamepadButton, GamepadStick,
    InputKeyCode, InputMouseButton, ModifierMask, PhysicalInput,
};
pub use raw_state::RawInputState;
pub use binding::Binding;
pub use action::ActionDef;
pub use action_map::ActionMap;
pub use system::InputSystem;
pub use serialize::{BindingOverrides, BindingPath};
pub use gamepad::{GamepadManager, GamepadInfo};
pub use gamepad_ui::GamepadUiEvent;
