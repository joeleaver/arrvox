//! RKP Engine — self-contained game engine for gaussian splat rendering.
//!
//! The engine runs on its own thread with its own tick loop. It owns the scene
//! (ECS world), renderer (RkpRenderer), physics, and animation. Communication
//! with the outside world (editor, game, test harness) is via:
//!
//! - **Commands in:** `crossbeam::channel::Sender<EngineCommand>` — async mutations
//! - **State out:** `StateCallback` — called each tick with current state
//! - **Pixels out:** `FrameCallback` — called each tick with rendered frame
//!
//! The engine never calls back into the client via traits. Callbacks are plain `Fn`.

pub mod animation;
pub mod behavior;
pub mod camera;
pub mod command;
pub mod components;
pub mod console;
pub mod environment;
pub mod component_registry;
pub mod file_watcher;
pub mod gameplay_loader;
pub mod import_profile;
pub mod import_worker;
pub mod play_mode;
pub mod procedural_snapshot;
pub mod gizmo;
pub mod inspector;
pub mod material_library;
pub mod project;
pub mod recent_projects;
pub mod scaffold;
pub mod scene_io;
pub mod snapshot;
pub mod engine;
pub mod scene_sync;
pub mod wireframe_builders;

pub use command::EngineCommand;
pub use snapshot::{StateUpdate, SceneObjectInfo, ModelInfo};
pub use material_library::MaterialInfo;
pub use engine::RkpEngine;

// Re-export the proc macros for use by gameplay crates.
pub use rkp_macros::rkp_component;
pub use rkp_macros::rkp_system;
