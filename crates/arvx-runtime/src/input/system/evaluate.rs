//! Per-frame `evaluate()`: walks the active action map, combines all binding
//! readings per action, applies dead-zone, and advances action-phase state machines.

use glam::Vec2;

use super::InputSystem;
use crate::input::types::{ActionPhase, ActionState};

impl InputSystem {
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
}
