//! `GeneratorOwned` — marker component on entities spawned by a generator.
//!
//! The regen path despawns every entity whose parent-in-the-marker equals
//! the generator entity that's about to re-run. This implements the
//! blow-away semantics agreed in the design: generator children are
//! disposable by default; opt-in persistence arrives via a slot-key
//! variant (M5).
//!
//! Not serialized — generator children re-materialize from the generator
//! function on scene load, so their parent pointer doesn't need to survive
//! across sessions. Manually registered as an engine built-in so the
//! inspector can enumerate it for debug purposes.

#[derive(Debug, Clone)]
pub struct GeneratorOwned {
    /// The generator entity that emitted this child.
    pub parent: hecs::Entity,
    /// The generator's `generation` counter at emit time. Bumped on
    /// each persistent-child reuse so an outside observer can tell
    /// when a child was last refreshed.
    pub generation: u64,
    /// Persistent slot key. `None` = anonymous child (blown away on
    /// every regen). `Some(key)` = persistent child (reused across
    /// regens by matching (parent, slot_key); only despawned when the
    /// generator stops emitting that key, or the parent goes away).
    /// Authored by the generator via
    /// `ctx.emit_persistent_child(slot_key, ...)`.
    pub slot_key: Option<String>,
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
        // Not serialized — generator children are re-emitted on load.
        serialize: |_, _| None,
        deserialize_insert: |_, _, _| Ok(()),
        on_add: None,
        on_remove: None,
    }
}
