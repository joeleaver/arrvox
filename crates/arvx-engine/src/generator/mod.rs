//! Generator system.
//!
//! Users write generator functions in `assets/scripts/generators/*.rs`
//! decorated with `#[arvx_generator(name = "x", params = XParams)]`. The engine
//! discovers them from the gameplay dylib, runs them on a worker thread in
//! response to param edits, and streams their output into the scene.
//!
//! M1 (current): macro + registry + state component only. No execution yet.
//! M2 adds the worker shell; M3 adds `voxelize_procedural` and `emit_child`.

pub mod context;
pub mod error;
pub mod owned;
pub mod preset;
pub mod registry;
pub mod state;
pub mod system;

pub use context::{child_cache_path, CancelToken, GeneratorContext, ProgressHandle};
pub use error::{GeneratorError, GeneratorStatus};
pub use owned::GeneratorOwned;
pub use preset::{GeneratorAssetConfig, GeneratorPresetInfo};
pub use registry::{
    CloneParamsFn, GenerateFn, GeneratorEntry, GeneratorRegistry, InsertDefaultParamsFn,
};
pub use state::GeneratorState;
pub use system::{GeneratorEvent, GeneratorSystem};

pub(crate) use owned::generator_owned_entry;
pub(crate) use state::generator_state_entry;
