//! System entry — metadata for a registered gameplay system.
//!
//! Each `#[rkp_system]` function generates an `inventory::submit!(SystemEntry { ... })`
//! so the engine can discover and schedule it from the gameplay dylib.

use super::Phase;

/// Metadata for a registered system, collected via `inventory`.
///
/// The `fn_ptr` is stored as `*const ()` because the proc macro crate cannot
/// depend on `SystemContext`. The executor transmutes it back to
/// `fn(&mut SystemContext)` at call time. This is safe if and only if every
/// registered system was originally a `fn(&mut SystemContext)` cast to
/// `*const ()`. The `#[rkp_system]` proc macro guarantees this.
pub struct SystemEntry {
    /// Function name (e.g., `"spin_system"`).
    pub name: &'static str,
    /// Module path for disambiguation.
    pub module_path: &'static str,
    /// Which phase this system runs in.
    pub phase: Phase,
    /// Systems that must run before this one (by name).
    pub after: &'static [&'static str],
    /// Systems that must run after this one (by name).
    pub before: &'static [&'static str],
    /// The system function pointer.
    pub fn_ptr: *const (),
}

// Safety: SystemEntry is only constructed from static data and fn pointers.
unsafe impl Send for SystemEntry {}
unsafe impl Sync for SystemEntry {}

impl std::fmt::Debug for SystemEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SystemEntry")
            .field("name", &self.name)
            .field("phase", &self.phase)
            .finish_non_exhaustive()
    }
}

inventory::collect!(SystemEntry);
