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
    pub sun_direction: [f32; 3],
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
            sun_direction: [-0.5, -0.8, -0.3],
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

    /// Build GPU shadow/AO params from these settings.
    pub fn to_shadow_ao_params(&self, num_objects: u32) -> rkp_render::rkp_shadow_ao::ShadowAoParams {
        // Negate sun_direction for light direction (sun points toward the light source,
        // but shadows trace away from it).
        let ld = self.sun_direction;
        let len = (ld[0] * ld[0] + ld[1] * ld[1] + ld[2] * ld[2]).sqrt().max(0.001);
        rkp_render::rkp_shadow_ao::ShadowAoParams {
            light_dir: [ld[0] / len, ld[1] / len, ld[2] / len],
            num_objects,
            light_intensity: self.sun_intensity,
            ao_radius: self.ao_radius,
            ao_steps: self.ao_steps,
            shadow_steps: self.shadow_steps,
        }
    }

    /// Build the default directional light GPU struct from these settings.
    pub fn to_gpu_light(&self) -> rkp_render::rkp_shade::GpuLight {
        let d = self.sun_direction;
        rkp_render::rkp_shade::GpuLight {
            position: [0.0, 0.0, 0.0, 0.0],
            color: [self.sun_color[0], self.sun_color[1], self.sun_color[2], self.sun_intensity],
            direction: [d[0], d[1], d[2], 0.0],
            params: [0.0; 4],
        }
    }
}
