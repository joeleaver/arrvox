//! Scene file format — .rkscene files.
//!
//! A scene contains objects with transforms, asset references,
//! camera state, and lights. Serialized as JSON.

use glam::Vec3;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A scene file — serialized to `.rkscene` as JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneFile {
    pub objects: Vec<SceneObject>,
    pub camera: CameraState,
    #[serde(default)]
    pub lights: Vec<SceneLight>,
    #[serde(default)]
    pub environment: Option<EnvironmentState>,
}

/// An object in the scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneObject {
    pub id: Uuid,
    pub name: String,
    pub position: [f32; 3],
    pub rotation: [f32; 3],
    pub scale: [f32; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<Uuid>,
    /// Path to the .rkp asset file (relative to project assets/).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset_path: Option<String>,
    /// Primitive type if this is an analytical object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primitive: Option<String>,
    #[serde(default)]
    pub material_id: u16,
    /// PointLight component data (if entity has one).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point_light: Option<ScenePointLight>,
    /// Camera component data (if entity has one).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<SceneCamera>,
    /// Generic component data — maps component name → JSON string.
    /// Used for gameplay components and any components not covered by
    /// the hardcoded fields above.
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub components: std::collections::HashMap<String, String>,
}

/// Saved PointLight component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenePointLight {
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
    #[serde(default = "default_true")]
    pub cast_shadow: bool,
}

fn default_true() -> bool { true }

/// Saved Camera component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneCamera {
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    pub active: bool,
}

/// Camera state stored in a scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraState {
    pub position: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub fov: f32,
}

impl Default for CameraState {
    fn default() -> Self {
        Self {
            position: [0.0, 2.0, 5.0],
            yaw: 0.0,
            pitch: 0.0,
            fov: 60.0,
        }
    }
}

/// A light in the scene (legacy format — new scenes use components on SceneObject).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneLight {
    pub id: Uuid,
    pub name: String,
    pub light_type: String,
    pub position: [f32; 3],
    pub intensity: f32,
    pub range: f32,
}

/// Saved environment settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentState {
    pub sky_color_top: [f32; 3],
    pub sky_color_horizon: [f32; 3],
    pub ambient_intensity: f32,
    pub sun_azimuth: f32,
    pub sun_elevation: f32,
    pub sun_color: [f32; 3],
    pub sun_intensity: f32,
    pub shadow_steps: u32,
    pub ao_radius: f32,
    pub ao_steps: u32,
    pub exposure: f32,
    // Volumetric fog (optional for backward compat with old scene files).
    #[serde(default)]
    pub fog_color: Option<[f32; 3]>,
    #[serde(default)]
    pub height_fog_density: Option<f32>,
    #[serde(default)]
    pub dust_density: Option<f32>,
    #[serde(default)]
    pub clouds_enabled: Option<bool>,
    #[serde(default)]
    pub cloud_altitude_min: Option<f32>,
    #[serde(default)]
    pub cloud_altitude_max: Option<f32>,
}

impl SceneFile {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            camera: CameraState::default(),
            lights: Vec::new(),
            environment: None,
        }
    }
}

impl EnvironmentState {
    pub fn from_settings(env: &crate::environment::EnvironmentSettings) -> Self {
        Self {
            sky_color_top: env.sky_color_top_override.unwrap_or([0.4, 0.6, 1.0]),
            sky_color_horizon: env.sky_color_horizon_override.unwrap_or([0.8, 0.85, 0.9]),
            ambient_intensity: env.ambient_intensity,
            sun_azimuth: env.sun_azimuth,
            sun_elevation: env.sun_elevation,
            sun_color: env.sun_color,
            sun_intensity: env.sun_intensity,
            shadow_steps: env.shadow_steps,
            ao_radius: env.ao_radius,
            ao_steps: env.ao_steps,
            exposure: env.exposure,
            fog_color: Some(env.fog_color),
            height_fog_density: Some(env.height_fog_density),
            dust_density: Some(env.dust_density),
            clouds_enabled: Some(env.clouds_enabled),
            cloud_altitude_min: Some(env.cloud_altitude_min),
            cloud_altitude_max: Some(env.cloud_altitude_max),
        }
    }

    pub fn to_settings(&self) -> crate::environment::EnvironmentSettings {
        let defaults = crate::environment::EnvironmentSettings::default();
        crate::environment::EnvironmentSettings {
            sky_color_top_override: Some(self.sky_color_top),
            sky_color_horizon_override: Some(self.sky_color_horizon),
            skip_sun_extinction: false,
            ambient_intensity: self.ambient_intensity,
            sun_azimuth: self.sun_azimuth,
            sun_elevation: self.sun_elevation,
            sun_color: self.sun_color,
            sun_intensity: self.sun_intensity,
            shadow_steps: self.shadow_steps,
            ao_radius: self.ao_radius,
            ao_steps: self.ao_steps,
            exposure: self.exposure,
            fog_color: self.fog_color.unwrap_or(defaults.fog_color),
            height_fog_density: self.height_fog_density.unwrap_or(defaults.height_fog_density),
            dust_density: self.dust_density.unwrap_or(defaults.dust_density),
            clouds_enabled: self.clouds_enabled.unwrap_or(defaults.clouds_enabled),
            cloud_altitude_min: self.cloud_altitude_min.unwrap_or(defaults.cloud_altitude_min),
            cloud_altitude_max: self.cloud_altitude_max.unwrap_or(defaults.cloud_altitude_max),
            ..defaults
        }
    }
}

/// Save a scene to disk.
pub fn save_scene(scene: &SceneFile, path: &std::path::Path) -> Result<(), String> {
    let json = serde_json::to_string_pretty(scene)
        .map_err(|e| format!("serialize scene: {e}"))?;
    std::fs::write(path, &json)
        .map_err(|e| format!("write scene: {e}"))?;
    eprintln!("[RkpEngine] saved scene to {}", path.display());
    Ok(())
}

/// Load a scene from disk.
pub fn load_scene(path: &std::path::Path) -> Result<SceneFile, String> {
    let json = std::fs::read_to_string(path)
        .map_err(|e| format!("read scene: {e}"))?;
    let scene: SceneFile = serde_json::from_str(&json)
        .map_err(|e| format!("parse scene: {e}"))?;
    eprintln!("[RkpEngine] loaded scene from {} ({} objects)", path.display(), scene.objects.len());
    Ok(scene)
}
