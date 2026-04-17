//! Action definitions — named input actions with bindings and dead zones.

use serde::{Serialize, Deserialize};

use super::binding::Binding;
use super::types::ControlType;

/// Definition of a named input action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionDef {
    /// Action name (e.g., "move", "jump", "fire").
    pub name: String,
    /// What kind of value this action produces.
    pub control_type: ControlType,
    /// Current bindings (may be remapped at runtime).
    pub bindings: Vec<Binding>,
    /// Default bindings stored at creation for reset.
    #[serde(skip)]
    pub default_bindings: Vec<Binding>,
    /// Dead zone threshold for analog values (default 0.1).
    pub dead_zone: f32,
}

impl ActionDef {
    /// Create a new action definition.
    pub fn new(name: impl Into<String>, control_type: ControlType, bindings: Vec<Binding>) -> Self {
        let name = name.into();
        let default_bindings = bindings.clone();
        Self {
            name,
            control_type,
            bindings,
            default_bindings,
            dead_zone: 0.1,
        }
    }

    /// Create with a custom dead zone.
    pub fn with_dead_zone(mut self, dead_zone: f32) -> Self {
        self.dead_zone = dead_zone;
        self
    }

    /// Reset bindings to defaults.
    pub fn reset_bindings(&mut self) {
        self.bindings = self.default_bindings.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::types::{InputKeyCode, PhysicalInput};

    #[test]
    fn action_def_new() {
        let action = ActionDef::new(
            "jump",
            ControlType::Digital,
            vec![Binding::simple(PhysicalInput::Key(InputKeyCode::Space))],
        );
        assert_eq!(action.name, "jump");
        assert_eq!(action.control_type, ControlType::Digital);
        assert_eq!(action.bindings.len(), 1);
        assert_eq!(action.default_bindings.len(), 1);
        assert_eq!(action.dead_zone, 0.1);
    }

    #[test]
    fn action_def_with_dead_zone() {
        let action = ActionDef::new("look", ControlType::Axis2D, vec![])
            .with_dead_zone(0.2);
        assert_eq!(action.dead_zone, 0.2);
    }

    #[test]
    fn action_def_reset_bindings() {
        let mut action = ActionDef::new(
            "fire",
            ControlType::Digital,
            vec![Binding::simple(PhysicalInput::Key(InputKeyCode::Space))],
        );
        // Modify bindings
        action.bindings.push(Binding::simple(PhysicalInput::Key(InputKeyCode::Enter)));
        assert_eq!(action.bindings.len(), 2);

        // Reset
        action.reset_bindings();
        assert_eq!(action.bindings.len(), 1);
    }
}
