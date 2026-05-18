//! Rebinding API: query/edit bindings on the active action map, find conflicts,
//! reset to defaults, and import/export per-map override deltas.

use std::collections::HashMap;

use super::{InputSystem, collect_inputs};
use crate::input::action_map::ActionMap;
use crate::input::binding::Binding;
use crate::input::types::PhysicalInput;

impl InputSystem {
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
