//! Environment settings — sky, lighting, shadows, AO, tone mapping.
//!
//! Stored in the engine, edited via commands, pushed to the renderer each frame.
//! Also pushed to the UI via StateUpdate for the environment panel.

/// All editable environment settings.
///
/// Directly serialized into scene files. `#[serde(default)]` means any field
/// missing from an older scene file takes its `Default` value, so adding new
/// fields here is forward-compatible and every existing field is saved without
/// an extra DTO to keep in sync.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
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
    /// Override sun surface color (None = atmosphere-computed from transmittance).
    pub sun_color_override: Option<[f32; 3]>,
    pub sun_intensity: f32,

    // ── Shadows ─────────────────────────────────────────────────────
    pub shadow_steps: u32,

    /// CSM near distance, in metres. The four cascades partition
    /// `[shadow_csm_near, shadow_csm_max_distance]`. NOT exposed in
    /// the editor UI — raising it actually *worsens* cascade 0's
    /// resolution because the slice [near, split1] then covers a
    /// wider mid-range chunk of the camera frustum (the bounding
    /// sphere grows with the *screen* footprint of the slice, not
    /// with the camera distance). Pinning to 0.1 m keeps cascade 0
    /// tight against the eye. Power users can override via
    /// `RKP_CSM_NEAR=...` for headless tuning.
    pub shadow_csm_near: f32,
    /// CSM (Cascaded Shadow Maps) far-distance cap, in metres. The four
    /// cascades partition `[shadow_csm_near, shadow_csm_max_distance]`.
    /// Anything beyond this distance is treated as fully lit (matches
    /// today's "out-of-bounds → 1.0" behaviour at the single-cascade
    /// edge).
    pub shadow_csm_max_distance: f32,
    /// PSSM hybrid factor. 0 = uniform (linear) splits across the
    /// full range (mid-range bias); 1 = log (geometric) splits
    /// (near-camera bias). With our `[0.1, 100]` m default range the
    /// uniform component dominates below λ ≈ 0.7, which is why the
    /// editor slider doesn't do much under that. Default 0.95 puts
    /// cascade 0 ≈ `[0.1, 1.79]` m → ~5 mm/texel at 1024². Drop
    /// toward 0.5 if mid-range shadows are visibly chunky; raise
    /// toward 1.0 for the sharpest near-camera detail at the cost
    /// of coarser far cascades.
    pub shadow_csm_lambda: f32,
    /// Additive depth bias applied to the surface depth before the
    /// shadow-map compare (light-space NDC z units). Positive pushes
    /// the surface toward the light → less self-shadowing at the cost
    /// of peter-panning. Replaces the previously-hardcoded 0.001.
    pub shadow_csm_depth_bias: f32,

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
    /// Elevation of the scene origin (Y=0) above sea level, in metres.
    /// Added to the camera's world Y to get the altitude fed into the
    /// atmosphere model (transmittance, sky-view, aerial perspective).
    /// Default 0 assumes Y=0 is ground/sea-level — set higher only for
    /// mountain scenes where Y=0 is already elevated.
    pub scene_elevation: f32,

    /// Linear-RGB albedo of the virtual "ground" that fills the sky-view
    /// LUT below the horizon. Gives empty voxel scenes a smooth
    /// earth-through-atmosphere fade instead of a black void at the edge
    /// of the ground plane. 0.3 grey is a reasonable desert/asphalt mix.
    pub ground_albedo: [f32; 3],

    // ── Volumetric fog ─────────────────────────────────────────────
    // Height-fog is the only participating-medium knob; distance haze is now
    // handled physically by the aerial-perspective LUT in the shade pass.
    pub fog_color: [f32; 3],
    pub height_fog_density: f32,
    pub fog_base_height: f32,
    pub fog_height_falloff: f32,
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
    /// When true, ray-march camera→sun through the cloud layer and attenuate
    /// direct sun contribution by the resulting optical depth. Cheap proxy for
    /// "clouds dim the world" without real cloud shadow maps.
    pub attenuate_sun_by_clouds: bool,

    // ── Cloud quality ──────────────────────────────────────────────
    /// Slab march sample count (main cost driver: linear in this).
    pub cloud_slab_steps: u32,
    /// Sun-shadow march samples per cloud sample (linear in this).
    pub cloud_shadow_steps: u32,
    /// Detail FBM octave count (finer cloud features).
    pub cloud_detail_octaves: u32,
    /// Multi-scatter octaves used inside clouds.
    pub cloud_ms_octaves: u32,
    /// Temporal accumulation weight for the current frame (0.1 = strong history,
    /// 0.5 = responsive but noisier).
    pub cloud_taa_alpha: f32,
}

impl Default for EnvironmentSettings {
    fn default() -> Self {
        Self {
            sky_color_top_override: None,
            sky_color_horizon_override: None,
            ambient_intensity: 1.0,
            sun_azimuth: 210.0,   // southwest
            sun_elevation: 45.0,  // mid-afternoon
            sun_color_override: None,
            sun_intensity: 110_000.0,
            shadow_steps: 32,
            shadow_csm_near: 0.1,
            shadow_csm_max_distance: 100.0,
            shadow_csm_lambda: 0.95,
            shadow_csm_depth_bias: 0.001,
            ao_radius: 0.1,
            ao_steps: 5,
            exposure: 0.0000254, // EV100=15 (sunny-16 rule): 1/(1.2 × 2^15)

            bloom_threshold: 1.0,
            bloom_knee: 0.5,
            bloom_intensity: 0.5,

            god_ray_density: 1.0,
            god_ray_weight: 0.01,
            god_ray_decay: 0.95,
            god_ray_exposure: 0.1,

            scene_elevation: 0.0,
            ground_albedo: [0.3, 0.3, 0.3],

            fog_color: [0.7, 0.75, 0.8],
            height_fog_density: 0.0,
            fog_base_height: 0.0,
            fog_height_falloff: 0.1,
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
            cloud_detail_weight: 0.5,
            cloud_weather_scale: 10000.0,
            cloud_wind_speed: 5.0,
            cloud_wind_dir: 0.0,
            attenuate_sun_by_clouds: true,
            // Defaults match the "High" preset.
            cloud_slab_steps: 32,
            cloud_shadow_steps: 4,
            cloud_detail_octaves: 4,
            cloud_ms_octaves: 3,
            cloud_taa_alpha: 0.25,
        }
    }
}

/// Named quality tiers that bundle the individual cloud render knobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloudQualityPreset {
    Low,
    Medium,
    High,
    Ultra,
}

/// (slab_steps, shadow_steps, detail_octaves, ms_octaves, taa_alpha)
pub fn cloud_quality_values(preset: CloudQualityPreset) -> (u32, u32, u32, u32, f32) {
    match preset {
        CloudQualityPreset::Low    => (16, 2, 2, 2, 0.15),
        CloudQualityPreset::Medium => (24, 3, 3, 3, 0.25),
        CloudQualityPreset::High   => (32, 4, 4, 3, 0.25),
        CloudQualityPreset::Ultra  => (48, 5, 5, 4, 0.35),
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

    /// Effective altitude of the camera above sea level, in metres.
    /// Adds the scene's elevation offset to the camera's world-space Y.
    pub fn effective_altitude(&self, cam_y: f32) -> f32 {
        self.scene_elevation + cam_y
    }

    /// Atmospheric-extinction-tinted sun colour (linear RGB), no intensity
    /// multiplier. Same shape fed to GpuLight / VolumetricParams; reusable
    /// by any pass that wants to render "sunlight" with a consistent hue.
    pub fn sun_tint(&self, cam_y: f32) -> [f32; 3] {
        let d = self.sun_direction();
        let sun_toward = [-d[0], -d[1], -d[2]];
        let trans = atmo::sun_transmittance(sun_toward, self.effective_altitude(cam_y));
        let atmo_color = [trans[0], 0.95 * trans[1], 0.9 * trans[2]];
        self.sun_color_override.unwrap_or(atmo_color)
    }

    /// Build GPU shade params from these settings.
    /// Sky colors and ambient are computed from the atmosphere model.
    /// Fixed atmosphere radiance (matches GPU ATMO_SUN_RADIANCE constant).
    /// Multi-scattering boost — must match MULTI_SCATTER_BOOST in rkp_shade.wgsl.
    pub fn to_shade_params(&self, cam_y: f32) -> rkp_render::rkp_shade::ShadeParams {
        let d = self.sun_direction();
        let sun_toward = [-d[0], -d[1], -d[2]];
        let altitude = self.effective_altitude(cam_y);
        // Sky colors for volumetric fog compatibility (GPU ambient uses LUT directly).
        let sky_top = atmo::sky([0.0, 1.0, 0.0], sun_toward, self.sun_intensity, altitude);
        let horizon_dir = {
            let el = 10.0f32.to_radians();
            [0.0, el.sin(), -el.cos()]
        };
        let sky_horizon = atmo::sky(horizon_dir, sun_toward, self.sun_intensity, altitude);

        rkp_render::rkp_shade::ShadeParams {
            num_lights: 1,
            ambient_intensity: self.ambient_intensity,
            camera_altitude: altitude,
            sun_intensity: self.sun_intensity,
            sky_color_top: self.sky_color_top_override.unwrap_or(sky_top),
            sky_color_horizon: self.sky_color_horizon_override.unwrap_or(sky_horizon),
            sun_dir: sun_toward,
            ambient_color: {
                let amb = atmo::ambient(sun_toward, self.sun_intensity, altitude);
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
        let altitude = self.effective_altitude(cam.position[1]);
        rkp_render::rkp_volumetric::VolumetricParams {
            cam_pos: cam.position,
            cam_forward: cam.forward,
            cam_right: cam.right,
            cam_up: cam.up,
            sun_dir: [sun_toward[0], sun_toward[1], sun_toward[2], 0.0],
            sun_color: {
                let trans = atmo::sun_transmittance(sun_toward, altitude);
                let base = self.sun_color_override.unwrap_or([1.0 * trans[0], 0.95 * trans[1], 0.9 * trans[2]]);
                [base[0] * self.sun_intensity, base[1] * self.sun_intensity, base[2] * self.sun_intensity, 0.0]
            },
            width: (width / 2).max(1),
            height: (height / 2).max(1),
            full_width: width,
            full_height: height,
            max_steps: self.vol_max_steps,
            step_size: self.vol_step_size,
            near: 0.5,
            far: self.vol_far,
            fog_color: [self.fog_color[0], self.fog_color[1], self.fog_color[2], 0.0],
            fog_height: [
                self.height_fog_density,
                self.fog_base_height,
                self.fog_height_falloff,
                0.0,
            ],
            frame_index,
            vol_ambient_r: {
                let a = atmo::ambient(sun_toward, self.sun_intensity, altitude);
                a[0] * self.ambient_intensity
            },
            vol_ambient_g: {
                let a = atmo::ambient(sun_toward, self.sun_intensity, altitude);
                a[1] * self.ambient_intensity
            },
            vol_ambient_b: {
                let a = atmo::ambient(sun_toward, self.sun_intensity, altitude);
                a[2] * self.ambient_intensity
            },
            prev_view_proj: cam.prev_vp,
        }
    }

    /// Build cloud params from settings.
    pub fn to_cloud_params(&self, time: f32) -> rkp_render::rkp_volumetric::CloudParams {
        let wind_rad = self.cloud_wind_dir.to_radians();
        rkp_render::rkp_volumetric::CloudParams {
            altitude: [
                self.cloud_altitude_min, self.cloud_altitude_max,
                // Convert coverage (0=clear, 1=overcast) to threshold. At coverage=0
                // the threshold sits above the typical peak of shape·weather·height_grad
                // so the sky genuinely clears; at coverage=1 it drops below so every
                // sample contributes.
                0.7 - 0.85 * self.cloud_coverage,
                self.cloud_density_scale,
            ],
            noise: [
                self.cloud_shape_freq, self.cloud_detail_freq,
                self.cloud_detail_weight, self.cloud_weather_scale,
            ],
            // Wind speed slider is in m/s, but the shape-noise wavelength is kilometres
            // so raw m/s produces imperceptible motion. Scale so a slider value of 1
            // looks like a gentle breeze over a minute of observation.
            wind: [wind_rad.sin(), wind_rad.cos(), self.cloud_wind_speed * 20.0, time],
            flags: [
                if self.clouds_enabled { 1.0 } else { 0.0 },
                self.cloud_coverage, // passed to shader to suppress weather variation at high coverage
                0.0, 0.0,
            ],
            quality: [
                self.cloud_slab_steps as f32,
                self.cloud_shadow_steps as f32,
                self.cloud_detail_octaves as f32,
                self.cloud_ms_octaves as f32,
            ],
            quality2: [self.cloud_taa_alpha, 0.0, 0.0, 0.0],
        }
    }

    /// Build the default directional light GPU struct from these settings.
    /// Sun color is attenuated by atmospheric extinction (orange at sunset, dark at night).
    pub fn to_gpu_light(&self, cam_y: f32) -> rkp_render::rkp_shade::GpuLight {
        let d = self.sun_direction();
        let sun_toward = [-d[0], -d[1], -d[2]];
        let trans = atmo::sun_transmittance(sun_toward, self.effective_altitude(cam_y));
        let atmo_color = [1.0 * trans[0], 0.95 * trans[1], 0.9 * trans[2]];
        let effective_color = self.sun_color_override.unwrap_or(atmo_color);
        rkp_render::rkp_shade::GpuLight {
            position: [0.0, 0.0, 0.0, 0.0],
            color: [effective_color[0], effective_color[1], effective_color[2], self.sun_intensity],
            direction: [d[0], d[1], d[2], 0.0],
            params: [0.0, 0.0, 0.0, 1.0], // w = cast_shadow
        }
    }
}
