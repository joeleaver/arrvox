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

pub mod camera;
pub mod command;
pub mod components;
pub mod file_watcher;
pub mod import_worker;
pub mod gizmo;
pub mod inspector;
pub mod project;
pub mod scene_io;
pub mod snapshot;
pub mod engine;
pub mod scene_sync;
pub mod wireframe_builders;

pub use command::EngineCommand;
pub use snapshot::{StateUpdate, SceneObjectInfo, ModelInfo};
pub use engine::RkpEngine;
