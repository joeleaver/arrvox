//! Generator registry — the catalog of registered generator functions.
//!
//! Generators register via `inventory::submit!` (emitted by the
//! `#[arvx_generator]` proc macro) and are collected into a `GeneratorRegistry`
//! either from the editor binary's own inventory or from the gameplay dylib.

use std::any::{Any, TypeId};
use std::collections::HashMap;

use super::context::GeneratorContext;
use super::error::GeneratorError;

/// Type-erased generator function pointer.
///
/// The proc macro generates a wrapper that downcasts the `&dyn Any` to the
/// concrete `XParams` type and calls the user's function. The function
/// returns `Ok(())` on clean completion; all output flows through context
/// methods (`emit_child` / `emit_persistent_child`).
///
/// The `for<'w>` HRTB is necessary because `GeneratorContext` carries a
/// lifetime tied to the worker's local borrows (evaluator, device, queue).
/// Without it we couldn't name the function as a `fn` pointer — only
/// worker-supplied contexts with concrete lifetimes could be passed in.
pub type GenerateFn =
    for<'w> fn(&dyn Any, &mut GeneratorContext<'w>) -> Result<(), GeneratorError>;

/// Clone the param component off a hecs entity into a boxed `dyn Any`.
/// The executor uses this to ship params onto the worker thread without
/// holding a borrow into the ECS world.
///
/// Returns `None` if the entity doesn't have the component.
pub type CloneParamsFn =
    fn(&hecs::World, hecs::Entity) -> Option<Box<dyn Any + Send>>;

/// Insert `XParams::default()` onto an entity. Used when spawning a new
/// generator so it has something for the inspector to render.
pub type InsertDefaultParamsFn =
    fn(&mut hecs::World, hecs::Entity);

/// One registered generator's metadata, submitted via `inventory`.
pub struct GeneratorEntry {
    /// Unique name for this generator (e.g. `"building"`).
    pub name: &'static str,
    /// Component name of the params struct (e.g. `"BuildingParams"`). Must
    /// match a registered `ComponentEntry` — the editor uses this to render
    /// the params inspector.
    pub param_component_name: &'static str,
    /// `TypeId` of the params struct. Used by the executor to sanity-check
    /// the downcast in the erased `GenerateFn` wrapper.
    pub param_type_id: TypeId,
    pub generate_fn: GenerateFn,
    pub clone_params: CloneParamsFn,
    pub insert_default_params: InsertDefaultParamsFn,
}

// Safety: all fields are `&'static` data, fn pointers, or `TypeId` — all of
// which are Send + Sync. `inventory::collect!` requires this bound.
unsafe impl Send for GeneratorEntry {}
unsafe impl Sync for GeneratorEntry {}

inventory::collect!(GeneratorEntry);

/// Catalog of all known generators, assembled from inventory + dylib.
pub struct GeneratorRegistry {
    /// Generators registered in the editor binary itself. None today; kept
    /// for symmetry with `ComponentRegistry` and easier unit tests.
    own_entries: Vec<&'static GeneratorEntry>,
    /// Generators from the hot-reloaded gameplay dylib.
    gameplay_entries: Vec<&'static GeneratorEntry>,
}

impl GeneratorRegistry {
    pub fn new() -> Self {
        let own_entries: Vec<&'static GeneratorEntry> =
            inventory::iter::<GeneratorEntry>.into_iter().collect();
        Self {
            own_entries,
            gameplay_entries: Vec::new(),
        }
    }

    /// Register a generator entry from the gameplay dylib.
    pub fn register_gameplay(&mut self, entry: &'static GeneratorEntry) {
        if !self.gameplay_entries.iter().any(|e| e.name == entry.name) {
            self.gameplay_entries.push(entry);
        }
    }

    /// Drop all gameplay entries (called before dylib unload).
    pub fn clear_gameplay(&mut self) {
        self.gameplay_entries.clear();
    }

    /// Look up by name. Gameplay entries shadow own entries of the same name.
    pub fn get(&self, name: &str) -> Option<&'static GeneratorEntry> {
        self.gameplay_entries
            .iter()
            .find(|e| e.name == name)
            .or_else(|| self.own_entries.iter().find(|e| e.name == name))
            .copied()
    }

    /// All known generator names, dedup'd.
    pub fn names(&self) -> Vec<&'static str> {
        let mut out: Vec<&'static str> = self
            .own_entries
            .iter()
            .chain(self.gameplay_entries.iter())
            .map(|e| e.name)
            .collect();
        out.sort();
        out.dedup();
        out
    }

    pub fn count(&self) -> usize {
        self.names().len()
    }

    pub fn has(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    /// Group entries by `param_component_name` for the editor's "insert
    /// default params" path after a generator is spawned.
    pub fn by_param_component(&self) -> HashMap<&'static str, &'static GeneratorEntry> {
        let mut m = HashMap::new();
        for e in self.own_entries.iter().chain(self.gameplay_entries.iter()) {
            m.insert(e.param_component_name, *e);
        }
        m
    }
}

impl Default for GeneratorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_registry_reports_nothing() {
        let reg = GeneratorRegistry {
            own_entries: Vec::new(),
            gameplay_entries: Vec::new(),
        };
        assert_eq!(reg.count(), 0);
        assert!(!reg.has("missing"));
        assert!(reg.get("missing").is_none());
        assert!(reg.names().is_empty());
    }

    fn leak(e: GeneratorEntry) -> &'static GeneratorEntry {
        Box::leak(Box::new(e))
    }

    fn fake_entry(name: &'static str, param_component: &'static str) -> GeneratorEntry {
        GeneratorEntry {
            name,
            param_component_name: param_component,
            param_type_id: std::any::TypeId::of::<()>(),
            generate_fn: |_, _| Ok(()),
            clone_params: |_, _| None,
            insert_default_params: |_, _| {},
        }
    }

    #[test]
    fn dedup_across_own_and_gameplay() {
        let own = leak(fake_entry("same", "OwnParams"));
        let game = leak(fake_entry("same", "GameParams"));

        let mut reg = GeneratorRegistry {
            own_entries: vec![own],
            gameplay_entries: Vec::new(),
        };
        assert_eq!(reg.count(), 1);
        reg.register_gameplay(game);
        assert_eq!(reg.count(), 1);
        // Gameplay shadows on lookup.
        assert_eq!(reg.get("same").unwrap().param_component_name, "GameParams");
    }

    #[test]
    fn register_gameplay_idempotent() {
        let e = leak(fake_entry("once", "P"));
        let mut reg = GeneratorRegistry {
            own_entries: Vec::new(),
            gameplay_entries: Vec::new(),
        };
        reg.register_gameplay(e);
        reg.register_gameplay(e);
        assert_eq!(reg.gameplay_entries.len(), 1);
    }

    #[test]
    fn clear_gameplay_keeps_own() {
        let own = leak(fake_entry("own", "P"));
        let game = leak(fake_entry("game", "Q"));
        let mut reg = GeneratorRegistry {
            own_entries: vec![own],
            gameplay_entries: vec![game],
        };
        assert_eq!(reg.count(), 2);
        reg.clear_gameplay();
        assert_eq!(reg.count(), 1);
        assert!(reg.has("own"));
        assert!(!reg.has("game"));
    }
}
