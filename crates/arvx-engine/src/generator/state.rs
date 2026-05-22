//! `GeneratorState` — the component that marks an entity as a generator.
//!
//! `generator_name` and `param_hash` persist. The remaining fields
//! (status, code_hash, generation) are transient and reset to defaults
//! after scene load. The persisted `param_hash` is what lets us SKIP
//! re-running a generator on every scene reopen: when the saved
//! children + their bake caches load and the freshly-computed param
//! hash matches the saved one, the system seeds its
//! `last_submitted_hash` from the saved value and submits nothing.

use serde::{Deserialize, Serialize};

use super::error::GeneratorStatus;

/// Component attached to entities that are generators.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GeneratorState {
    /// Name of the registered generator (e.g. `"building"`).
    pub generator_name: String,

    /// Current lifecycle status. Transient. The deserialiser
    /// reconstructs it post-load via [`Self::initial_status_after_load`]
    /// based on whether `param_hash` is non-zero (i.e. whether this
    /// entity has ever completed a run before).
    #[serde(skip)]
    pub status: GeneratorStatus,

    /// Hash of the params at the most recent successful run. Persisted
    /// across scene reload so the generator system can recognise
    /// "params unchanged since last bake" and skip a redundant run —
    /// a session that opens a saved scene then immediately closes it
    /// should cost zero generator CPU.
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
    /// Compute the post-load `status`. `Ready` if we've ever generated
    /// (saved `param_hash` non-zero); `Pending` otherwise. The system's
    /// scan_and_submit tick re-derives the actual current `param_hash`
    /// from the live params and only re-runs if it differs.
    pub fn initial_status_after_load(&self) -> GeneratorStatus {
        if self.param_hash != 0 {
            GeneratorStatus::Ready
        } else {
            GeneratorStatus::Pending
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
        field_visible: None,
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
    fn serde_roundtrip_persists_name_and_param_hash() {
        let mut s = GeneratorState::new("rock");
        s.status = GeneratorStatus::Ready;
        s.param_hash = 1234;
        s.code_hash = 5678;
        s.generation = 7;

        let json = serde_json::to_string(&s).unwrap();
        let back: GeneratorState = serde_json::from_str(&json).unwrap();

        assert_eq!(back.generator_name, "rock");
        // status / code_hash / generation reset (transient).
        assert_eq!(back.status, GeneratorStatus::Pending);
        assert_eq!(back.code_hash, 0);
        assert_eq!(back.generation, 0);
        // param_hash persists so the system can detect unchanged params
        // on reload and skip a redundant regen.
        assert_eq!(back.param_hash, 1234);
        assert_eq!(back.initial_status_after_load(), GeneratorStatus::Ready);
    }

    #[test]
    fn initial_status_pending_when_never_generated() {
        let s = GeneratorState::new("never-run");
        assert_eq!(s.param_hash, 0);
        assert_eq!(s.initial_status_after_load(), GeneratorStatus::Pending);
    }
}
