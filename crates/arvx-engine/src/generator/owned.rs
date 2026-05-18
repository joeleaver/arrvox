//! `GeneratorOwned` — marker component on entities spawned by a generator.
//!
//! Every emitted child carries a stable `slot_key` (assigned by the
//! generator at emit time). Regen reuses existing children in place by
//! matching `(parent_uuid, slot_key)`; children whose key disappears
//! in a later generation are despawned. Slot keys also key the
//! on-disk bake cache, which is what makes save/reload skip regen.
//!
//! Fully serialised so children + their bake caches survive a scene
//! reload without forcing the generator to re-run. The runtime engine
//! map looks up the parent entity by UUID at query time — there's no
//! transient `parent: hecs::Entity` field (Entity ids don't survive
//! process restart, and a stale `Entity::DANGLING` placeholder between
//! load and a fixup pass invited bugs).

use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorOwned {
    /// UUID of the generator entity that emitted this child. Map to
    /// the runtime `hecs::Entity` via `EngineState::entity_uuids`.
    pub parent_uuid: Uuid,
    /// The generator's `generation` counter at emit time. Bumped on
    /// each reuse. Transient — recomputed from the generator's current
    /// state on each regen.
    #[serde(skip)]
    pub generation: u64,
    /// Stable identity assigned by the generator at emit time. The
    /// engine matches `(parent_uuid, slot_key)` to find an existing
    /// child to reuse on regen; absent matches → spawn fresh.
    /// Children whose key the generator stops emitting are despawned.
    pub slot_key: String,
}

// ─── ComponentEntry for the built-in registry ──────────────────────────

use crate::component_registry::{ComponentEntry, FieldMeta};

static GENERATOR_OWNED_FIELDS: [FieldMeta; 0] = [];

pub(crate) fn generator_owned_entry() -> ComponentEntry {
    ComponentEntry {
        name: "GeneratorOwned",
        meta: &GENERATOR_OWNED_FIELDS,
        mandatory: false,
        has: |world, entity| world.get::<&GeneratorOwned>(entity).is_ok(),
        get_field: |_, _, _| Err("GeneratorOwned exposes no editable fields".into()),
        set_field: |_, _, _, _| Err("GeneratorOwned is not editable".into()),
        add_default: |_, _| Err("GeneratorOwned is set by the generator system".into()),
        remove: |world, entity| {
            world
                .remove_one::<GeneratorOwned>(entity)
                .map(|_| ())
                .map_err(|e| format!("{e}"))
        },
        // Persisted now: parent_uuid + slot_key round-trip via serde.
        // `generation` resets to 0 on load (transient), and the
        // generator system rebumps it the next time it emits.
        serialize: |world, entity| {
            let c = world.get::<&GeneratorOwned>(entity).ok()?;
            serde_json::to_string(&*c).ok()
        },
        deserialize_insert: |world, entity, json| {
            let c: GeneratorOwned = serde_json::from_str(json).map_err(|e| format!("{e}"))?;
            world.insert_one(entity, c).map_err(|e| format!("{e}"))
        },
        on_add: None,
        on_remove: None,
    }
}
