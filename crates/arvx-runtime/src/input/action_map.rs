//! Action maps — named groups of actions (e.g., "gameplay", "menu", "vehicle").

use serde::{Serialize, Deserialize};

use super::action::ActionDef;

/// A named group of input actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionMap {
    /// Map name (e.g., "gameplay", "menu").
    pub name: String,
    /// Actions in this map.
    pub actions: Vec<ActionDef>,
}

impl ActionMap {
    /// Create a new action map.
    pub fn new(name: impl Into<String>, actions: Vec<ActionDef>) -> Self {
        Self {
            name: name.into(),
            actions,
        }
    }

    /// Find an action by name.
    pub fn find_action(&self, name: &str) -> Option<&ActionDef> {
        self.actions.iter().find(|a| a.name == name)
    }

    /// Find an action by name (mutable).
    pub fn find_action_mut(&mut self, name: &str) -> Option<&mut ActionDef> {
        self.actions.iter_mut().find(|a| a.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::binding::Binding;
    use crate::input::types::*;

    fn sample_map() -> ActionMap {
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
        ])
    }

    #[test]
    fn find_action_by_name() {
        let map = sample_map();
        assert!(map.find_action("jump").is_some());
        assert!(map.find_action("move").is_some());
        assert!(map.find_action("nonexistent").is_none());
    }

    #[test]
    fn find_action_mut_modify() {
        let mut map = sample_map();
        let action = map.find_action_mut("jump").unwrap();
        action.dead_zone = 0.5;
        assert_eq!(map.find_action("jump").unwrap().dead_zone, 0.5);
    }
}
