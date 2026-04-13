//! Example gameplay components.
//!
//! Add your own components here using `#[rkp_component]`.

use rkp_engine::rkp_component;
use serde::{Deserialize, Serialize};

/// Health component — tracks current and maximum hit points.
#[rkp_component]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Health {
    #[range(0.0, 1000.0)]
    pub current: f32,
    #[range(0.0, 1000.0)]
    pub max: f32,
}

/// Spin component — rotates the object around an axis each frame.
#[rkp_component]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spin {
    /// Rotation speed in degrees per second.
    #[range(0.0, 720.0)]
    pub speed: f32,
    /// Rotation axis (normalized).
    pub axis: glam::Vec3,
}

impl Default for Spin {
    fn default() -> Self {
        Self {
            speed: 45.0,
            axis: glam::Vec3::Y,
        }
    }
}

/// Collectible marker — this object can be picked up.
#[rkp_component]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Collectible {
    /// Score value when collected.
    #[range(0.0, 10000.0)]
    pub value: f32,
    /// Whether this has already been collected.
    pub collected: bool,
}
