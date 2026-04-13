//! RKIPatch gameplay components — user-defined components for game logic.
//!
//! This crate compiles as a cdylib for hot-reload. The engine loads it at startup
//! and reloads when the file changes (after recompilation).
//!
//! Define components using `#[rkp_component]`:
//!
//! ```ignore
//! use rkp_engine::rkp_component;
//!
//! #[rkp_component]
//! #[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
//! pub struct Health {
//!     #[range(0.0, 1000.0)]
//!     pub current: f32,
//!     #[range(0.0, 1000.0)]
//!     pub max: f32,
//! }
//! ```

pub mod components;

use rkp_engine::component_registry::ComponentEntry;

/// Exported function called by the engine to discover gameplay components.
///
/// Returns a list of ComponentEntry references. The engine registers them
/// and can serialize/deserialize/inspect them generically.
///
/// # Safety
/// Called across FFI boundary. The returned entries contain function pointers
/// into this dylib — they become invalid if the dylib is unloaded.
#[unsafe(no_mangle)]
pub extern "C" fn rkp_gameplay_entries() -> GameplayEntries {
    // Collect all inventory-registered entries from this dylib.
    let entries: Vec<&'static ComponentEntry> =
        inventory::iter::<ComponentEntry>.into_iter().collect();
    GameplayEntries {
        ptr: entries.as_ptr(),
        len: entries.len(),
        _storage: entries,
    }
}

/// FFI-safe container for gameplay component entries.
#[repr(C)]
pub struct GameplayEntries {
    pub ptr: *const &'static ComponentEntry,
    pub len: usize,
    // Keep the Vec alive so ptr stays valid.
    _storage: Vec<&'static ComponentEntry>,
}
