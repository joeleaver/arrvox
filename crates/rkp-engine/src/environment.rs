//! Environment settings — sky, lighting, shadows, AO, tone mapping.
//!
//! Stored in the engine, edited via commands, pushed to the renderer each frame.
//! Also pushed to the UI via StateUpdate for the environment panel.

/// All editable environment settings.
#[derive(Debug, Clone, PartialEq)]
pub struct EnvironmentSettings {
    // ── Sky ──────────────────────────────────────────────────────────
    pub sky_color_top: [f32; 3],
    pub sky_color_horizon: [f32; 3],
    pub ambient_intensity: f32,

    // ── Sun / directional light ─────────────────────────────────────
    /// Sun azimuth in degrees (0 = North, 90 = East, 180 = South, 270 = West).
    pub sun_azimuth: f32,
    /// Sun elevation in degrees (0 = horizon, 90 = directly overhead, negative = below).
    pub sun_elevation: f32,
    pub sun_color: [f32; 3],
    pub sun_intensity: f32,

    // ── Shadows ─────────────────────────────────────────────────────
    pub shadow_steps: u32,

    // ── Ambient occlusion ───────────────────────────────────────────
    pub ao_radius: f32,
    pub ao_steps: u32,

    // ── Tone mapping ────────────────────────────────────────────────
    pub exposure: f32,
}

impl Default for EnvironmentSettings {
    fn default() -> Self {
        Self {
            sky_color_top: [0.4, 0.6, 1.0],
            sky_color_horizon: [0.8, 0.85, 0.9],
            ambient_intensity: 0.3,
            sun_azimuth: 210.0,   // southwest
            sun_elevation: 45.0,  // mid-afternoon
            sun_color: [1.0, 0.95, 0.9],
            sun_intensity: 2.0,
            shadow_steps: 32,
            ao_radius: 0.1,
            ao_steps: 5,
            exposure: 1.0,
        }
    }
}

impl EnvironmentSettings {
    /// Compute the direction light shines (FROM sky TOWARD ground).
    /// Y-up: +X = East, +Z = South, +Y = Up.
    /// Azimuth 0 = North (−Z), 90 = East (+X), 180 = South (+Z), 270 = West (−X).
    pub fn sun_direction(&self) -> [f32; 3] {
        let az = self.sun_azimuth.to_radians();
        let el = self.sun_elevation.to_radians();
        let cos_el = el.cos();
        [
            -(az.sin() * cos_el),
            -(el.sin()),
            -(az.cos() * cos_el),
        ]
    }

    /// Build GPU shade params from these settings.
    pub fn to_shade_params(&self) -> rkp_render::rkp_shade::ShadeParams {
        rkp_render::rkp_shade::ShadeParams {
            num_lights: 1,
            ambient_intensity: self.ambient_intensity,
            sky_color_top: self.sky_color_top,
            sky_color_horizon: self.sky_color_horizon,
            ..Default::default()
        }
    }

    /// Normalized direction FROM surface TOWARD the light source.
    /// This is the negated sun_direction (shadow rays trace toward the light).
    pub fn light_dir_normalized(&self) -> [f32; 3] {
        let d = self.sun_direction();
        [-d[0], -d[1], -d[2]]
    }

    /// Build the default directional light GPU struct from these settings.
    pub fn to_gpu_light(&self) -> rkp_render::rkp_shade::GpuLight {
        let d = self.sun_direction();
        rkp_render::rkp_shade::GpuLight {
            position: [0.0, 0.0, 0.0, 0.0],
            color: [self.sun_color[0], self.sun_color[1], self.sun_color[2], self.sun_intensity],
            direction: [d[0], d[1], d[2], 0.0],
            params: [0.0; 4],
        }
    }
}
