//! RKIPatch gameplay crate — DEPRECATED.
//!
//! Gameplay code no longer lives in the engine workspace. Instead, the engine
//! scaffolds a per-project gameplay crate from `assets/scripts/{components,systems}/`
//! in the project directory. See `rkp-engine/src/scaffold.rs`.
//!
//! This crate is kept in the workspace to avoid breaking the build but contains
//! no code. It will be removed in a future cleanup.
