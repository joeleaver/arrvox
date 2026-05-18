//! # arvx-physics
//!
//! Physics integration for the Arrvox engine.
//!
//! Rapier handles rigid body dynamics, but world collision uses a custom
//! adapter that evaluates the SDF at contact sample points rather than
//! relying on mesh colliders.
//!
//! This crate provides:
//! - Rapier physics world integration and stepping
//! - Custom SDF collision adapter (SDF evaluation at contact points)
//! - Character controller (capsule-vs-SDF, iterative slide)
//! - Collision shape generation from SDF data

#![warn(missing_docs)]

pub use rapier3d;
pub mod rapier_world;
pub mod sdf_collision;
pub mod rigid_body;
pub mod character_controller;
pub mod destruction;
