//! Game store — key-value state + event bus for cross-system communication.
//!
//! Systems read and write named values. Events are emitted during a frame
//! and drained at the end of the LateUpdate phase.

use std::collections::HashMap;

/// Dynamically-typed value for the game store.
#[derive(Debug, Clone, PartialEq)]
pub enum GameValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Vec3([f32; 3]),
}

/// An event emitted by a system, visible to other systems in the same frame.
#[derive(Debug, Clone)]
pub struct StoreEvent {
    /// Event name (e.g., "enemy_died", "item_collected").
    pub name: String,
    /// Which entity emitted this event, if any.
    pub source: Option<hecs::Entity>,
    /// Optional payload.
    pub data: Option<GameValue>,
}

/// Key-value store for game state + per-frame event buffer.
pub struct GameStore {
    values: HashMap<String, GameValue>,
    events: Vec<StoreEvent>,
}

impl GameStore {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
            events: Vec::new(),
        }
    }

    // ── Values ──────────────────────────────────────────────────────

    /// Set a value by key.
    pub fn set(&mut self, key: &str, value: GameValue) {
        self.values.insert(key.to_owned(), value);
    }

    /// Get a value by key.
    pub fn get(&self, key: &str) -> Option<&GameValue> {
        self.values.get(key)
    }

    /// Remove a value by key.
    pub fn remove(&mut self, key: &str) {
        self.values.remove(key);
    }

    /// Get a float value, returning 0.0 if missing or wrong type.
    pub fn get_float(&self, key: &str) -> f64 {
        match self.values.get(key) {
            Some(GameValue::Float(v)) => *v,
            _ => 0.0,
        }
    }

    /// Get a bool value, returning false if missing or wrong type.
    pub fn get_bool(&self, key: &str) -> bool {
        match self.values.get(key) {
            Some(GameValue::Bool(v)) => *v,
            _ => false,
        }
    }

    /// Get a string value, returning empty string if missing or wrong type.
    pub fn get_string(&self, key: &str) -> &str {
        match self.values.get(key) {
            Some(GameValue::String(v)) => v.as_str(),
            _ => "",
        }
    }

    // ── Events ──────────────────────────────────────────────────────

    /// Emit an event visible to other systems this frame.
    pub fn emit(&mut self, name: impl Into<String>, source: Option<hecs::Entity>, data: Option<GameValue>) {
        self.events.push(StoreEvent {
            name: name.into(),
            source,
            data,
        });
    }

    /// Iterate events matching a name.
    pub fn events(&self, name: &str) -> impl Iterator<Item = &StoreEvent> {
        self.events.iter().filter(move |e| e.name == name)
    }

    /// Clear all events. Called at the end of each frame.
    pub fn drain_events(&mut self) {
        self.events.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_get_values() {
        let mut store = GameStore::new();
        store.set("health", GameValue::Float(100.0));
        store.set("alive", GameValue::Bool(true));
        assert_eq!(store.get_float("health"), 100.0);
        assert_eq!(store.get_bool("alive"), true);
        assert_eq!(store.get_float("missing"), 0.0);
    }

    #[test]
    fn events_emit_and_drain() {
        let mut store = GameStore::new();
        store.emit("hit", None, Some(GameValue::Float(25.0)));
        store.emit("hit", None, Some(GameValue::Float(10.0)));
        store.emit("heal", None, None);

        assert_eq!(store.events("hit").count(), 2);
        assert_eq!(store.events("heal").count(), 1);
        assert_eq!(store.events("miss").count(), 0);

        store.drain_events();
        assert_eq!(store.events("hit").count(), 0);
    }
}
