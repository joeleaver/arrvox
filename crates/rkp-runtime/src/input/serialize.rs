//! RON serialization and binding overrides for input configuration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use super::action_map::ActionMap;
use super::binding::Binding;

/// Binding overrides — only stores the delta from defaults.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BindingOverrides {
    /// Map name -> (action name -> new bindings).
    pub maps: HashMap<String, HashMap<String, Vec<Binding>>>,
}

/// Path into a composite binding for per-leg rebinding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindingPath {
    /// The entire binding.
    Root,
    /// A named part of a composite (e.g., "up", "down", "left", "right", "positive", "negative").
    Part(String),
}

/// Load action maps from a RON file.
pub fn load_action_maps(path: &Path) -> anyhow::Result<Vec<ActionMap>> {
    let contents = std::fs::read_to_string(path)?;
    let maps: Vec<ActionMap> = ron::from_str(&contents)?;
    Ok(maps)
}

/// Save action maps to a RON file.
pub fn save_action_maps(path: &Path, maps: &[ActionMap]) -> anyhow::Result<()> {
    let pretty = ron::ser::PrettyConfig::default();
    let s = ron::ser::to_string_pretty(maps, pretty)?;
    std::fs::write(path, s)?;
    Ok(())
}

/// Save binding overrides to a RON file.
pub fn save_overrides(path: &Path, overrides: &BindingOverrides) -> anyhow::Result<()> {
    let pretty = ron::ser::PrettyConfig::default();
    let s = ron::ser::to_string_pretty(overrides, pretty)?;
    std::fs::write(path, s)?;
    Ok(())
}

/// Load binding overrides from a RON file.
pub fn load_overrides(path: &Path) -> anyhow::Result<BindingOverrides> {
    let contents = std::fs::read_to_string(path)?;
    let overrides: BindingOverrides = ron::from_str(&contents)?;
    Ok(overrides)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::action::ActionDef;
    use crate::input::types::*;

    #[test]
    fn ron_roundtrip_simple_binding() {
        let binding = Binding::simple(PhysicalInput::Key(InputKeyCode::W));
        let s = ron::to_string(&binding).unwrap();
        let back: Binding = ron::from_str(&s).unwrap();
        assert_eq!(binding, back);
    }

    #[test]
    fn ron_roundtrip_composite_2d() {
        let binding = Binding::Composite2D {
            up: PhysicalInput::Key(InputKeyCode::W),
            down: PhysicalInput::Key(InputKeyCode::S),
            left: PhysicalInput::Key(InputKeyCode::A),
            right: PhysicalInput::Key(InputKeyCode::D),
        };
        let s = ron::to_string(&binding).unwrap();
        let back: Binding = ron::from_str(&s).unwrap();
        assert_eq!(binding, back);
    }

    #[test]
    fn ron_roundtrip_action_map() {
        let map = ActionMap::new(
            "gameplay",
            vec![
                ActionDef::new(
                    "jump",
                    ControlType::Digital,
                    vec![Binding::simple(PhysicalInput::Key(InputKeyCode::Space))],
                ),
                ActionDef::new(
                    "move",
                    ControlType::Axis2D,
                    vec![Binding::Composite2D {
                        up: PhysicalInput::Key(InputKeyCode::W),
                        down: PhysicalInput::Key(InputKeyCode::S),
                        left: PhysicalInput::Key(InputKeyCode::A),
                        right: PhysicalInput::Key(InputKeyCode::D),
                    }],
                ),
            ],
        );
        let s = ron::to_string(&map).unwrap();
        let back: ActionMap = ron::from_str(&s).unwrap();
        assert_eq!(back.name, "gameplay");
        assert_eq!(back.actions.len(), 2);
        assert_eq!(back.actions[0].name, "jump");
        assert_eq!(back.actions[1].name, "move");
        assert_eq!(back.actions[0].bindings.len(), 1);
    }

    #[test]
    fn ron_roundtrip_binding_overrides() {
        let mut overrides = BindingOverrides::default();
        let mut action_overrides = HashMap::new();
        action_overrides.insert(
            "jump".to_string(),
            vec![Binding::simple(PhysicalInput::Key(InputKeyCode::Enter))],
        );
        overrides.maps.insert("gameplay".to_string(), action_overrides);

        let s = ron::to_string(&overrides).unwrap();
        let back: BindingOverrides = ron::from_str(&s).unwrap();
        assert_eq!(back.maps.len(), 1);
        let gameplay = back.maps.get("gameplay").unwrap();
        let jump = gameplay.get("jump").unwrap();
        assert_eq!(jump.len(), 1);
        assert_eq!(jump[0], Binding::simple(PhysicalInput::Key(InputKeyCode::Enter)));
    }

    #[test]
    fn file_roundtrip_action_maps() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("maps.ron");

        let maps = vec![ActionMap::new(
            "test",
            vec![ActionDef::new(
                "fire",
                ControlType::Digital,
                vec![Binding::simple(PhysicalInput::Key(InputKeyCode::Space))],
            )],
        )];

        save_action_maps(&path, &maps).unwrap();
        let loaded = load_action_maps(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].name, "test");
        assert_eq!(loaded[0].actions[0].name, "fire");
    }

    #[test]
    fn file_roundtrip_overrides() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("overrides.ron");

        let mut overrides = BindingOverrides::default();
        let mut actions = HashMap::new();
        actions.insert(
            "jump".to_string(),
            vec![
                Binding::simple(PhysicalInput::Key(InputKeyCode::Space)),
                Binding::simple(PhysicalInput::GamepadButton(GamepadButton::South)),
            ],
        );
        overrides.maps.insert("gameplay".to_string(), actions);

        save_overrides(&path, &overrides).unwrap();
        let loaded = load_overrides(&path).unwrap();
        let gameplay = loaded.maps.get("gameplay").unwrap();
        let jump = gameplay.get("jump").unwrap();
        assert_eq!(jump.len(), 2);
    }

    #[test]
    fn ron_roundtrip_composite_axis() {
        let binding = Binding::CompositeAxis {
            positive: PhysicalInput::Key(InputKeyCode::D),
            negative: PhysicalInput::Key(InputKeyCode::A),
        };
        let s = ron::to_string(&binding).unwrap();
        let back: Binding = ron::from_str(&s).unwrap();
        assert_eq!(binding, back);
    }

    #[test]
    fn ron_roundtrip_gamepad_inputs() {
        let inputs = vec![
            Binding::simple(PhysicalInput::GamepadButton(GamepadButton::South)),
            Binding::simple(PhysicalInput::GamepadAxis(GamepadAxis::LeftTrigger)),
            Binding::simple(PhysicalInput::GamepadStick(GamepadStick::Left)),
            Binding::simple(PhysicalInput::MouseButton(InputMouseButton::Right)),
            Binding::simple(PhysicalInput::MouseDelta),
            Binding::simple(PhysicalInput::ScrollWheel),
        ];
        for binding in &inputs {
            let s = ron::to_string(binding).unwrap();
            let back: Binding = ron::from_str(&s).unwrap();
            assert_eq!(*binding, back);
        }
    }

    #[test]
    fn default_bindings_skipped_in_serialization() {
        let action = ActionDef::new(
            "jump",
            ControlType::Digital,
            vec![Binding::simple(PhysicalInput::Key(InputKeyCode::Space))],
        );
        assert_eq!(action.default_bindings.len(), 1);

        let s = ron::to_string(&action).unwrap();
        let back: ActionDef = ron::from_str(&s).unwrap();
        // default_bindings is skipped, so it should be empty after deserialization
        assert!(back.default_bindings.is_empty());
        // But bindings should be preserved
        assert_eq!(back.bindings.len(), 1);
    }

    #[test]
    fn parse_sample_game_default_map() {
        let contents = include_str!("../../../../assets/input_maps/game_default.ron");
        let maps: Vec<ActionMap> = ron::from_str(contents).unwrap();
        assert_eq!(maps.len(), 2);
        assert_eq!(maps[0].name, "gameplay");
        assert_eq!(maps[0].actions.len(), 10);
        assert_eq!(maps[1].name, "menu");
        assert_eq!(maps[1].actions.len(), 3);
    }
}
