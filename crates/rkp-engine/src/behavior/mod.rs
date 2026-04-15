//! Behavior system — phased execution of gameplay systems.
//!
//! The developer writes `#[rkp_system]` functions in their gameplay crate.
//! The engine discovers them via `inventory` from the gameplay dylib and
//! runs them each frame during play mode.
//!
//! # Phases
//!
//! Systems run in three phases per frame:
//! 1. **Update** — gameplay, AI, input (variable dt)
//! 2. **FixedUpdate** — physics-rate logic (fixed dt)
//! 3. *(engine steps physics)*
//! 4. **LateUpdate** — camera follow, cleanup (variable dt)
//!
//! Commands (spawn/despawn/insert/remove) are flushed between each phase.

pub mod phase;
pub mod system_entry;
pub mod game_store;
pub mod engine_access;
pub mod command_queue;
pub mod system_context;
pub mod scheduler;
pub mod executor;

pub use phase::Phase;
pub use system_entry::SystemEntry;
pub use system_context::SystemContext;
pub use command_queue::{CommandQueue, TempEntity, ViewportRequest};
pub use game_store::{GameStore, GameValue, StoreEvent};
pub use engine_access::{EngineAccess, TransformUpdate};
pub use executor::BehaviorExecutor;
pub use scheduler::ScheduleError;
