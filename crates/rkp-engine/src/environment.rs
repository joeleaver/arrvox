//! Environment settings — sky, lighting, shadows, AO, tone mapping.
//!
//! Stored in the engine, edited via commands, pushed to the renderer each frame.
//! Also pushed to the UI via StateUpdate for the environment panel.

/// All editable environment settings.
#[derive(Debug, Clone, PartialEq)]
pub struct EnvironmentSettings {
    // ── Sky / Atmosphere ──────────────────────────────────────────────
    /// Override sky top color (None = computed from atmosphere model).
    pub sky_color_top_override: Option<[f32; 3]>,
    /// Override sky horizon color (None = computed from atmosphere model).
    pub sky_color_horizon_override: Option<[f32; 3]>,
    pub ambient_intensity: f32,

    // ── Sun / directional light ─────────────────────────────────────
    pub sun_azimuth: f32,
    pub sun_elevation: f32,
    /// Sun color. When `skip_sun_extinction` is false, atmosphere applies
    /// transmittance on top (orange at sunset). When true, used directly.
    pub sun_color: [f32; 3],
    pub skip_sun_extinction: bool,
    pub sun_intensity: f32,

    // ── Shadows ─────────────────────────────────────────────────────
    pub shadow_steps: u32,

    // ── Ambient occlusion ───────────────────────────────────────────
    pub ao_radius: f32,
    pub ao_steps: u32,

    // ── Tone mapping ────────────────────────────────────────────────
    pub exposure: f32,

    // ── Bloom ───────────────────────────────────────────────────────
    pub bloom_threshold: f32,
    pub bloom_knee: f32,
    pub bloom_intensity: f32,

    // ── God Rays ────────────────────────────────────────────────────
    pub god_ray_density: f32,
    pub god_ray_weight: f32,
    pub god_ray_decay: f32,
    pub god_ray_exposure: f32,

    // ── Atmosphere ─────────────────────────────────────────────────
    pub camera_altitude: f32,

    // ── Volumetric fog ─────────────────────────────────────────────
    pub fog_color: [f32; 3],
    pub height_fog_density: f32,
    pub fog_base_height: f32,
    pub fog_height_falloff: f32,
    pub distance_fog_density: f32,
    pub distance_fog_falloff: f32,
    pub dust_density: f32,
    pub dust_asymmetry: f32,
    pub vol_max_steps: u32,
    pub vol_step_size: f32,
    pub vol_far: f32,

    // ── Procedural clouds ──────────────────────────────────────────
    pub clouds_enabled: bool,
    pub cloud_altitude_min: f32,
    pub cloud_altitude_max: f32,
    /// Cloud coverage: 0 = clear sky, 1 = full overcast.
    pub cloud_coverage: f32,
    pub cloud_density_scale: f32,
    pub cloud_shape_freq: f32,
    pub cloud_detail_freq: f32,
    pub cloud_detail_weight: f32,
    pub cloud_weather_scale: f32,
    pub cloud_wind_speed: f32,
    pub cloud_wind_dir: f32,
}

impl Default for EnvironmentSettings {
    fn default() -> Self {
        Self {
            sky_color_top_override: None,
            sky_color_horizon_override: None,
            ambient_intensity: 1.0,
            sun_azimuth: 210.0,   // southwest
            sun_elevation: 45.0,  // mid-afternoon
            sun_color: [1.0, 0.95, 0.9],
            skip_sun_extinction: false,
            sun_intensity: 110_000.0,
            shadow_steps: 32,
            ao_radius: 0.1,
            ao_steps: 5,
            exposure: 0.0000254, // EV100=15 (sunny-16 rule): 1/(1.2 × 2^15)

            bloom_threshold: 1.0,
            bloom_knee: 0.5,
            bloom_intensity: 0.5,

            god_ray_density: 1.0,
            god_ray_weight: 0.01,
            god_ray_decay: 0.97,
            god_ray_exposure: 0.3,

            camera_altitude: 100.0,

            fog_color: [0.7, 0.75, 0.8],
            height_fog_density: 0.0,
            fog_base_height: 0.0,
            fog_height_falloff: 0.1,
            distance_fog_density: 0.0,
            distance_fog_falloff: 0.005,
            dust_density: 0.0,
            dust_asymmetry: 0.3,
            vol_max_steps: 32,
            vol_step_size: 2.0,
            vol_far: 200.0,

            clouds_enabled: false,
            cloud_altitude_min: 1000.0,
            cloud_altitude_max: 3000.0,
            cloud_coverage: 0.5,
            cloud_density_scale: 1.0,
            cloud_shape_freq: 0.0003,
            cloud_detail_freq: 0.002,
            cloud_detail_weight: 0.3,
            cloud_weather_scale: 10000.0,
            cloud_wind_speed: 5.0,
            cloud_wind_dir: 0.0,
        }
    }
}

// --- CPU-side atmospheric scattering (mirrors the GPU shader) ---

mod atmo {
    // Must match constants in rkp_shade.wgsl (Bruneton 2017 / Hillaire 2020).
    const EARTH_R: f32 = 6_360_000.0;
    const ATMO_R: f32 = 6_460_000.0;
    const RAYLEIGH_H: f32 = 8_000.0;
    const MIE_H: f32 = 1_200.0;
    const BETA_R: [f32; 3] = [5.802e-6, 13.558e-6, 33.1e-6];
    const BETA_M: f32 = 3.996e-6;

    fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 { a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }
    fn len3(a: [f32; 3]) -> f32 { dot3(a, a).sqrt() }
    fn scale3(a: [f32; 3], s: f32) -> [f32; 3] { [a[0]*s, a[1]*s, a[2]*s] }
    fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] { [a[0]+b[0], a[1]+b[1], a[2]+b[2]] }
    fn mul3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] { [a[0]*b[0], a[1]*b[1], a[2]*b[2]] }

    fn ray_sphere(origin: [f32; 3], dir: [f32; 3], radius: f32) -> Option<(f32, f32)> {
        let b = dot3(origin, dir);
        let c = dot3(origin, origin) - radius * radius;
        let d = b * b - c;
        if d < 0.0 { return None; }
        let s = d.sqrt();
        Some((-b - s, -b + s))
    }

    /// Compute atmospheric scattering for a single view direction.
    pub fn sky(ray_dir: [f32; 3], sun_dir: [f32; 3], sun_intensity: f32, cam_alt: f32) -> [f32; 3] {
        let origin = [0.0, EARTH_R + cam_alt, 0.0];
        let atmo = match ray_sphere(origin, ray_dir, ATMO_R) {
            Some((_, far)) if far > 0.0 => far,
            _ => return [0.0; 3],
        };
        let t_start = 0.0f32;
        let mut t_end = atmo;
        if let Some((near, _)) = ray_sphere(origin, ray_dir, EARTH_R) {
            if near > 0.0 { t_end = t_end.min(near); }
        }

        let cos_sun = dot3(ray_dir, sun_dir);
        let phase_r = (3.0 / (16.0 * std::f32::consts::PI)) * (1.0 + cos_sun * cos_sun);
        let g = 0.8f32;
        let g2 = g * g;
        let denom = 1.0 + g2 - 2.0 * g * cos_sun;
        let phase_m = (1.0 - g2) / (4.0 * std::f32::consts::PI * denom.max(1e-6).powf(1.5));

        let steps = 16;
        let step_len = (t_end - t_start) / steps as f32;
        let mut od_r = [0.0f32; 3];
        let mut od_m = [0.0f32; 3];
        let mut scatter = [0.0f32; 3];

        for i in 0..steps {
            let t = t_start + (i as f32 + 0.5) * step_len;
            let pos = add3(origin, scale3(ray_dir, t));
            let alt = len3(pos) - EARTH_R;

            let dr = (-alt / RAYLEIGH_H).exp() * step_len;
            let dm = (-alt / MIE_H).exp() * step_len;
            od_r = add3(od_r, scale3(BETA_R, dr));
            od_m = add3(od_m, [BETA_M * dm; 3]);

            // Secondary sun march.
            let sun_far = match ray_sphere(pos, sun_dir, ATMO_R) {
                Some((_, f)) => f,
                None => continue,
            };
            let sun_steps = 4;
            let ss_len = sun_far / sun_steps as f32;
            let mut od_sr = [0.0f32; 3];
            let mut od_sm = [0.0f32; 3];
            let mut shadowed = false;
            for j in 0..sun_steps {
                let st = (j as f32 + 0.5) * ss_len;
                let sp = add3(pos, scale3(sun_dir, st));
                let sa = len3(sp) - EARTH_R;
                if sa < 0.0 { shadowed = true; break; }
                let sdr = (-sa / RAYLEIGH_H).exp() * ss_len;
                let sdm = (-sa / MIE_H).exp() * ss_len;
                od_sr = add3(od_sr, scale3(BETA_R, sdr));
                od_sm = add3(od_sm, [BETA_M * sdm; 3]);
            }
            if shadowed { continue; }

            let total_od = [
                od_r[0] + od_m[0] + od_sr[0] + od_sm[0],
                od_r[1] + od_m[1] + od_sr[1] + od_sm[1],
                od_r[2] + od_m[2] + od_sr[2] + od_sm[2],
            ];
            let trans = [(-total_od[0]).exp(), (-total_od[1]).exp(), (-total_od[2]).exp()];
            for c in 0..3 {
                scatter[c] += (dr * BETA_R[c] * phase_r + dm * BETA_M * phase_m)
                            * trans[c] * sun_intensity;
            }
        }
        scatter
    }

    /// Compute sun transmittance (extinction along sun direction from camera).
    pub fn sun_transmittance(sun_dir: [f32; 3], cam_alt: f32) -> [f32; 3] {
        let origin = [0.0, EARTH_R + cam_alt, 0.0];
        let sun_far = match ray_sphere(origin, sun_dir, ATMO_R) {
            Some((_, f)) if f > 0.0 => f,
            _ => return [0.0; 3],
        };
        let steps = 8;
        let step_len = sun_far / steps as f32;
        let mut od_r = [0.0f32; 3];
        let mut od_m = [0.0f32; 3];
        for i in 0..steps {
            let t = (i as f32 + 0.5) * step_len;
            let pos = add3(origin, scale3(sun_dir, t));
            let alt = len3(pos) - EARTH_R;
            if alt < 0.0 { return [0.0; 3]; }
            let dr = (-alt / RAYLEIGH_H).exp() * step_len;
            let dm = (-alt / MIE_H).exp() * step_len;
            od_r = add3(od_r, scale3(BETA_R, dr));
            od_m = add3(od_m, [BETA_M * dm; 3]);
        }
        [
            (-(od_r[0] + od_m[0])).exp(),
            (-(od_r[1] + od_m[1])).exp(),
            (-(od_r[2] + od_m[2])).exp(),
        ]
    }

    /// Compute hemisphere-average ambient sky irradiance.
    pub fn ambient(sun_dir: [f32; 3], sun_intensity: f32, cam_alt: f32) -> [f32; 3] {
        // Sample 6 directions and weight.
        let dirs: [[f32; 3]; 6] = [
            [0.0, 1.0, 0.0],                                    // up
            [0.707, 0.5, 0.0],   [-0.707, 0.5, 0.0],           // horizon-ish E/W
            [0.0, 0.5, 0.707],   [0.0, 0.5, -0.707],           // horizon-ish N/S
            [0.0, 0.3, 0.0],                                    // low
        ];
        let weights = [0.3, 0.15, 0.15, 0.15, 0.15, 0.1];
        let mut result = [0.0f32; 3];
        for i in 0..6 {
            let d = dirs[i];
            let len = len3(d);
            let norm = [d[0]/len, d[1]/len, d[2]/len];
            let s = sky(norm, sun_dir, sun_intensity, cam_alt);
            for c in 0..3 { result[c] += s[c] * weights[i]; }
        }
        result
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
    /// Sky colors and ambient are computed from the atmosphere model.
    /// Fixed atmosphere radiance (matches GPU ATMO_SUN_RADIANCE constant).
    /// Multi-scattering boost — must match MULTI_SCATTER_BOOST in rkp_shade.wgsl.
    pub fn to_shade_params(&self) -> rkp_render::rkp_shade::ShadeParams {
        let d = self.sun_direction();
        let sun_toward = [-d[0], -d[1], -d[2]];
        // Sky colors for volumetric fog compatibility (GPU ambient uses LUT directly).
        let sky_top = atmo::sky([0.0, 1.0, 0.0], sun_toward, self.sun_intensity, self.camera_altitude);
        let horizon_dir = {
            let el = 10.0f32.to_radians();
            [0.0, el.sin(), -el.cos()]
        };
        let sky_horizon = atmo::sky(horizon_dir, sun_toward, self.sun_intensity, self.camera_altitude);

        rkp_render::rkp_shade::ShadeParams {
            num_lights: 1,
            ambient_intensity: self.ambient_intensity,
            camera_altitude: self.camera_altitude,
            sun_intensity: self.sun_intensity,
            sky_color_top: self.sky_color_top_override.unwrap_or(sky_top),
            sky_color_horizon: self.sky_color_horizon_override.unwrap_or(sky_horizon),
            sun_dir: sun_toward,
            ambient_color: {
                let amb = atmo::ambient(sun_toward, self.sun_intensity, self.camera_altitude);
                [amb[0] * self.ambient_intensity, amb[1] * self.ambient_intensity, amb[2] * self.ambient_intensity]
            },
            ..Default::default()
        }
    }

    /// Build volumetric params from settings + camera + resolution.
    pub fn to_volumetric_params(
        &self,
        cam: &rkp_render::rkp_scene::CameraUniforms,
        width: u32,
        height: u32,
        frame_index: u32,
    ) -> rkp_render::rkp_volumetric::VolumetricParams {
        let d = self.sun_direction();
        let sun_toward = [-d[0], -d[1], -d[2]]; // toward sun (negated)
        rkp_render::rkp_volumetric::VolumetricParams {
            cam_pos: cam.position,
            cam_forward: cam.forward,
            cam_right: cam.right,
            cam_up: cam.up,
            sun_dir: [sun_toward[0], sun_toward[1], sun_toward[2], 0.0],
            sun_color: {
                let trans = atmo::sun_transmittance(sun_toward, self.camera_altitude);
                // Use PBR sun intensity for fog/cloud lighting (not atmosphere scale).
                [
                    self.sun_color[0] * trans[0] * self.sun_intensity,
                    self.sun_color[1] * trans[1] * self.sun_intensity,
                    self.sun_color[2] * trans[2] * self.sun_intensity,
                    0.0,
                ]
            },
            width: (width / 2).max(1),
            height: (height / 2).max(1),
            full_width: width,
            full_height: height,
            max_steps: self.vol_max_steps,
            step_size: self.vol_step_size,
            near: 0.5,
            far: self.vol_far,
            fog_color: [
                self.fog_color[0], self.fog_color[1], self.fog_color[2],
                if self.height_fog_density > 0.0 { 1.0 } else { 0.0 },
            ],
            fog_height: [
                self.height_fog_density, self.fog_base_height, self.fog_height_falloff,
                if self.distance_fog_density > 0.0 { 1.0 } else { 0.0 },
            ],
            fog_distance: [
                self.distance_fog_density, self.distance_fog_falloff,
                self.dust_density, self.dust_asymmetry,
            ],
            frame_index,
            vol_ambient_r: {
                let a = atmo::ambient(sun_toward, self.sun_intensity, self.camera_altitude);
                a[0] * self.ambient_intensity
            },
            vol_ambient_g: {
                let a = atmo::ambient(sun_toward, self.sun_intensity, self.camera_altitude);
                a[1] * self.ambient_intensity
            },
            vol_ambient_b: {
                let a = atmo::ambient(sun_toward, self.sun_intensity, self.camera_altitude);
                a[2] * self.ambient_intensity
            },
        }
    }

    /// Build cloud params from settings.
    pub fn to_cloud_params(&self, time: f32) -> rkp_render::rkp_volumetric::CloudParams {
        let wind_rad = self.cloud_wind_dir.to_radians();
        rkp_render::rkp_volumetric::CloudParams {
            altitude: [
                self.cloud_altitude_min, self.cloud_altitude_max,
                // Convert coverage (0=clear, 1=overcast) to threshold.
                // At coverage=0: threshold=0.3 (clips most noise = clear).
                // At coverage=1: threshold=-0.05 (adds to base = full overcast).
                0.3 - 0.35 * self.cloud_coverage,
                self.cloud_density_scale,
            ],
            noise: [
                self.cloud_shape_freq, self.cloud_detail_freq,
                self.cloud_detail_weight, self.cloud_weather_scale,
            ],
            wind: [wind_rad.sin(), wind_rad.cos(), self.cloud_wind_speed, time],
            flags: [
                if self.clouds_enabled { 1.0 } else { 0.0 },
                self.cloud_coverage, // passed to shader to suppress weather variation at high coverage
                0.0, 0.0,
            ],
        }
    }

    /// Build the default directional light GPU struct from these settings.
    /// Sun color is attenuated by atmospheric extinction (orange at sunset, dark at night).
    pub fn to_gpu_light(&self) -> rkp_render::rkp_shade::GpuLight {
        let d = self.sun_direction();
        let sun_toward = [-d[0], -d[1], -d[2]];
        let trans = atmo::sun_transmittance(sun_toward, self.camera_altitude);
        let effective_color = if self.skip_sun_extinction {
            self.sun_color
        } else {
            [self.sun_color[0] * trans[0], self.sun_color[1] * trans[1], self.sun_color[2] * trans[2]]
        };
        rkp_render::rkp_shade::GpuLight {
            position: [0.0, 0.0, 0.0, 0.0],
            color: [effective_color[0], effective_color[1], effective_color[2], self.sun_intensity],
            direction: [d[0], d[1], d[2], 0.0],
            params: [0.0, 0.0, 0.0, 1.0], // w = cast_shadow
        }
    }
}
