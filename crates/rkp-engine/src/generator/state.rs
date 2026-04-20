//! `GeneratorState` — the component that marks an entity as a generator.
//!
//! Only `generator_name` is persisted. Everything else (status, param_hash,
//! code_hash, generation) is transient: it resets to defaults after scene load
//! so generators re-run on reopen.

use serde::{Deserialize, Serialize};

use super::error::GeneratorStatus;

/// Component attached to entities that are generators.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GeneratorState {
    /// Name of the registered generator (e.g. `"building"`).
    pub generator_name: String,

    /// Current lifecycle status. Transient — resets to `Pending` on load so
    /// every generator re-runs after scene reopen.
    #[serde(skip)]
    pub status: GeneratorStatus,

    /// Hash of the current params. Executor compares against the hash it saw
    /// at submission time to detect stale results. Transient.
    #[serde(skip)]
    pub param_hash: u64,

    /// Hash of the gameplay dylib at last successful run. Changes on
    /// hot-reload; triggers a re-run. Transient.
    #[serde(skip)]
    pub code_hash: u64,

    /// Monotonic counter — bumped each time a run completes. The context
    /// exposes this so generators can derive deterministic seeds that vary
    /// across re-runs (useful for "give me a new random result" buttons).
    /// Transient.
    #[serde(skip)]
    pub generation: u64,
}

impl Default for GeneratorState {
    fn default() -> Self {
        Self {
            generator_name: String::new(),
            status: GeneratorStatus::Pending,
            param_hash: 0,
            code_hash: 0,
            generation: 0,
        }
    }
}

impl GeneratorState {
    pub fn new(generator_name: impl Into<String>) -> Self {
        Self {
            generator_name: generator_name.into(),
            ..Default::default()
        }
    }
}

// ─── ComponentEntry for the built-in registry ──────────────────────────

use crate::component_registry::{ComponentEntry, FieldMeta};
use crate::inspector::{FieldType, FieldValue};

static GENERATOR_STATE_FIELDS: [FieldMeta; 1] = [FieldMeta {
    name: "generator_name",
    field_type: FieldType::String,
    range: None,
    transient: false,
    struct_fields: None,
    asset_filter: None,
    enum_options: None,
    scrub: false,
}];

pub(crate) fn generator_state_entry() -> ComponentEntry {
    ComponentEntry {
        name: "GeneratorState",
        meta: &GENERATOR_STATE_FIELDS,
        mandatory: false,
        has: |world, entity| world.get::<&GeneratorState>(entity).is_ok(),
        get_field: |world, entity, field| {
            let c = world
                .get::<&GeneratorState>(entity)
                .map_err(|_| "no GeneratorState".to_string())?;
            match field {
                "generator_name" => Ok(FieldValue::String(c.generator_name.clone())),
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        set_field: |world, entity, field, value| {
            let mut c = world
                .get::<&mut GeneratorState>(entity)
                .map_err(|_| "no GeneratorState".to_string())?;
            match field {
                "generator_name" => {
                    if let FieldValue::String(s) = value {
                        c.generator_name = s;
                        Ok(())
                    } else {
                        Err("type mismatch".into())
                    }
                }
                _ => Err(format!("unknown field '{field}'")),
            }
        },
        add_default: |world, entity| {
            world
                .insert_one(entity, GeneratorState::default())
                .map_err(|e| format!("{e}"))
        },
        remove: |world, entity| {
            world
                .remove_one::<GeneratorState>(entity)
                .map(|_| ())
                .map_err(|e| format!("{e}"))
        },
        serialize: |world, entity| {
            let c = world.get::<&GeneratorState>(entity).ok()?;
            serde_json::to_string(&*c).ok()
        },
        deserialize_insert: |world, entity, json| {
            let c: GeneratorState = serde_json::from_str(json).map_err(|e| format!("{e}"))?;
            world.insert_one(entity, c).map_err(|e| format!("{e}"))
        },
        on_add: None,
        on_remove: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_pending_with_name() {
        let s = GeneratorState::new("tree");
        assert_eq!(s.generator_name, "tree");
        assert_eq!(s.status, GeneratorStatus::Pending);
        assert_eq!(s.param_hash, 0);
        assert_eq!(s.code_hash, 0);
        assert_eq!(s.generation, 0);
    }

    #[test]
    fn serde_roundtrip_persists_name_only() {
        let mut s = GeneratorState::new("rock");
        s.status = GeneratorStatus::Ready;
        s.param_hash = 1234;
        s.code_hash = 5678;
        s.generation = 7;

        let json = serde_json::to_string(&s).unwrap();
        let back: GeneratorState = serde_json::from_str(&json).unwrap();

        assert_eq!(back.generator_name, "rock");
        assert_eq!(back.status, GeneratorStatus::Pending);
        assert_eq!(back.param_hash, 0);
        assert_eq!(back.code_hash, 0);
        assert_eq!(back.generation, 0);
    }
}
