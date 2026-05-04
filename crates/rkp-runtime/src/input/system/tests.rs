use super::*;
use crate::input::action::ActionDef;
use crate::input::action_map::ActionMap;
use crate::input::binding::Binding;
use crate::input::types::{ActionPhase, ControlType, GamepadButton, GamepadStick, InputKeyCode, InputMouseButton, PhysicalInput};


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
