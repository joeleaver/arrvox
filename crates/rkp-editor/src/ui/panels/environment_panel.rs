//! Environment panel — sky, lighting, shadows, AO, and tone mapping settings.

use std::rc::Rc;

use rinch::prelude::*;

use crate::CommandSender;
use crate::ui::store::EditorStore;
use super::prop_controls::*;

type CmdSignal = Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>;

#[component]
pub fn EnvironmentPanel() -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd_tx: CmdSignal = Signal::new(use_context::<CommandSender>().0);

    let env = store.environment.get();

    let ambient = Signal::new(env.ambient_intensity);
    let camera_altitude = Signal::new(env.camera_altitude);

    let sky_top_override_on = Signal::new(env.sky_color_top_override.is_some());
    let sky_top_color = Signal::new({
        let c = env.sky_color_top_override.unwrap_or([0.4, 0.6, 1.0]);
        [c[0], c[1], c[2], 1.0]
    });
    let sky_horizon_override_on = Signal::new(env.sky_color_horizon_override.is_some());
    let sky_horizon_color = Signal::new({
        let c = env.sky_color_horizon_override.unwrap_or([0.8, 0.85, 0.9]);
        [c[0], c[1], c[2], 1.0]
    });

    let sun_azimuth = Signal::new(env.sun_azimuth);
    let sun_elevation = Signal::new(env.sun_elevation);
    let sun_color_override_on = Signal::new(env.sun_color_override.is_some());
    let sun_color_override = Signal::new({
        let c = env.sun_color_override.unwrap_or([1.0, 0.95, 0.9]);
        [c[0], c[1], c[2], 1.0]
    });
    let sun_intensity = Signal::new(env.sun_intensity);

    let shadow_steps = Signal::new(env.shadow_steps as f32);
    let ao_radius = Signal::new(env.ao_radius);
    let ao_steps = Signal::new(env.ao_steps as f32);

    let exposure = Signal::new(env.exposure);

    let bloom_threshold = Signal::new(env.bloom_threshold);
    let bloom_knee = Signal::new(env.bloom_knee);
    let bloom_intensity = Signal::new(env.bloom_intensity);

    let god_ray_density = Signal::new(env.god_ray_density);
    let god_ray_exposure = Signal::new(env.god_ray_exposure);
    let god_ray_decay = Signal::new(env.god_ray_decay);

    let dust_density = Signal::new(env.dust_density);
    let dust_asymmetry = Signal::new(env.dust_asymmetry);
    let height_fog_density = Signal::new(env.height_fog_density);
    let fog_base_height = Signal::new(env.fog_base_height);
    let fog_height_falloff = Signal::new(env.fog_height_falloff);
    let distance_fog_density = Signal::new(env.distance_fog_density);
    let fog_color = Signal::new([env.fog_color[0], env.fog_color[1], env.fog_color[2], 1.0]);
    let vol_far = Signal::new(env.vol_far);

    let clouds_enabled = Signal::new(env.clouds_enabled);
    let cloud_altitude_min = Signal::new(env.cloud_altitude_min);
    let cloud_altitude_max = Signal::new(env.cloud_altitude_max);
    let cloud_coverage = Signal::new(env.cloud_coverage);
    let cloud_density_scale = Signal::new(env.cloud_density_scale);
    let cloud_wind_speed = Signal::new(env.cloud_wind_speed);
    let cloud_wind_dir = Signal::new(env.cloud_wind_dir);

    let sky_collapsed = Signal::new(false);
    let light_collapsed = Signal::new(false);
    let shadow_collapsed = Signal::new(false);
    let tone_collapsed = Signal::new(false);
    let bloom_collapsed = Signal::new(false);
    let god_ray_collapsed = Signal::new(false);
    let fog_collapsed = Signal::new(false);
    let cloud_collapsed = Signal::new(true);

    let env_f32 = move |field: &'static str| -> Rc<dyn Fn(f32)> {
        Rc::new(move |v: f32| {
            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::UpdateEnvironment {
                field: field.into(),
                value: v.to_string(),
            });
        })
    };
    let env_bool = move |field: &'static str| -> Rc<dyn Fn(bool)> {
        Rc::new(move |v: bool| {
            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::UpdateEnvironment {
                field: field.into(),
                value: if v { "true".into() } else { "false".into() },
            });
        })
    };
    let env_color3 = move |field: &'static str| -> Rc<dyn Fn([f32; 4])> {
        Rc::new(move |v: [f32; 4]| {
            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::UpdateEnvironment {
                field: field.into(),
                value: format!("[{},{},{}]", v[0], v[1], v[2]),
            });
        })
    };

    rsx! {
        div {
            style: "display:flex;flex-direction:column;height:100%;overflow-y:auto;\
                    color:#ccc;font-size:12px;",

            // ── Atmosphere section ────────────────────────────────────
            {prop_section_header(__scope, "Atmosphere", sky_collapsed, None)}
            if !sky_collapsed.get() {
                div {
                    style: "padding:6px 12px;display:flex;flex-direction:column;gap:4px;",
                    {prop_slider(__scope, "Ambient", ambient, 0.0, 5.0, 0.1, env_f32("ambient_intensity"))}
                    {prop_slider(__scope, "Camera Altitude (m)", camera_altitude, 0.0, 5000.0, 10.0, env_f32("camera_altitude"))}

                    // Override: sky top color
                    {prop_checkbox(__scope, "Override Sky Top", sky_top_override_on, {
                        let tx = cmd_tx;
                        Rc::new(move |v: bool| {
                            if !v {
                                let _ = tx.get().send(rkp_engine::EngineCommand::UpdateEnvironment {
                                    field: "sky_color_top_override_enabled".into(), value: "false".into(),
                                });
                            }
                        })
                    })}
                    if sky_top_override_on.get() {
                        {prop_color(__scope, "Sky Top", sky_top_color, env_color3("sky_color_top_override"))}
                    }

                    // Override: sky horizon color
                    {prop_checkbox(__scope, "Override Sky Horizon", sky_horizon_override_on, {
                        let tx = cmd_tx;
                        Rc::new(move |v: bool| {
                            if !v {
                                let _ = tx.get().send(rkp_engine::EngineCommand::UpdateEnvironment {
                                    field: "sky_color_horizon_override_enabled".into(), value: "false".into(),
                                });
                            }
                        })
                    })}
                    if sky_horizon_override_on.get() {
                        {prop_color(__scope, "Sky Horizon", sky_horizon_color, env_color3("sky_color_horizon_override"))}
                    }
                }
            }

            // ── Sun / Lighting section ───────────────────────────────
            {prop_section_header(__scope, "Sun", light_collapsed, None)}
            if !light_collapsed.get() {
                div {
                    style: "padding:6px 12px;display:flex;flex-direction:column;gap:4px;",
                    {prop_slider(__scope, "Intensity (lux)", sun_intensity, 0.0, 200000.0, 1000.0, env_f32("sun_intensity"))}

                    // Override: sun color (bypasses atmosphere extinction)
                    {prop_checkbox(__scope, "Override Sun Color", sun_color_override_on, {
                        let tx = cmd_tx;
                        Rc::new(move |v: bool| {
                            if !v {
                                let _ = tx.get().send(rkp_engine::EngineCommand::UpdateEnvironment {
                                    field: "sun_color_override_enabled".into(), value: "false".into(),
                                });
                            }
                        })
                    })}
                    if sun_color_override_on.get() {
                        {prop_color(__scope, "Sun Color", sun_color_override, env_color3("sun_color_override"))}
                    }

                    // Sun direction widget: azimuth + elevation
                    {sun_direction_widget(__scope, sun_azimuth, sun_elevation, cmd_tx)}
                }
            }

            // ── Shadows & AO section ─────────────────────────────────
            {prop_section_header(__scope, "Shadows & AO", shadow_collapsed, None)}
            if !shadow_collapsed.get() {
                div {
                    style: "padding:6px 12px;display:flex;flex-direction:column;gap:4px;",
                    {prop_slider(__scope, "Shadow Steps", shadow_steps, 4.0, 64.0, 4.0, env_f32("shadow_steps"))}
                    {prop_slider(__scope, "AO Radius", ao_radius, 0.01, 1.0, 0.01, env_f32("ao_radius"))}
                    {prop_slider(__scope, "AO Steps", ao_steps, 1.0, 16.0, 1.0, env_f32("ao_steps"))}
                }
            }

            // ── Tone Mapping section ─────────────────────────────────
            {prop_section_header(__scope, "Tone Mapping", tone_collapsed, None)}
            if !tone_collapsed.get() {
                div {
                    style: "padding:6px 12px;display:flex;flex-direction:column;gap:4px;",
                    {prop_scrub(
                        __scope,
                        "Exposure",
                        Memo::new(move || exposure.get()),
                        0.000001, 0.01, 0.000001,
                        {
                            let base = env_f32("exposure");
                            Rc::new(move |v: f32| {
                                exposure.set(v);
                                base(v);
                            })
                        },
                    )}
                }
            }

            // ── Bloom section ────────────────────────────────────────────
            {prop_section_header(__scope, "Bloom", bloom_collapsed, None)}
            if !bloom_collapsed.get() {
                div {
                    style: "padding:6px 12px;display:flex;flex-direction:column;gap:4px;",
                    {prop_slider(__scope, "Threshold", bloom_threshold, 0.0, 5.0, 0.1, env_f32("bloom_threshold"))}
                    {prop_slider(__scope, "Knee", bloom_knee, 0.0, 1.0, 0.01, env_f32("bloom_knee"))}
                    {prop_slider(__scope, "Intensity", bloom_intensity, 0.0, 2.0, 0.01, env_f32("bloom_intensity"))}
                }
            }

            // ── God Rays section ─────────────────────────────────────────
            {prop_section_header(__scope, "God Rays", god_ray_collapsed, None)}
            if !god_ray_collapsed.get() {
                div {
                    style: "padding:6px 12px;display:flex;flex-direction:column;gap:4px;",
                    {prop_slider(__scope, "Density", god_ray_density, 0.0, 3.0, 0.1, env_f32("god_ray_density"))}
                    {prop_slider(__scope, "Exposure", god_ray_exposure, 0.0, 1.0, 0.01, env_f32("god_ray_exposure"))}
                    {prop_slider(__scope, "Decay", god_ray_decay, 0.9, 1.0, 0.005, env_f32("god_ray_decay"))}
                }
            }

            // ── Fog section ─────────────────────────────────────────────
            {prop_section_header(__scope, "Fog", fog_collapsed, None)}
            if !fog_collapsed.get() {
                div {
                    style: "padding:6px 12px;display:flex;flex-direction:column;gap:4px;",
                    {prop_color(__scope, "Fog Color", fog_color, env_color3("fog_color"))}
                    {prop_slider(__scope, "Dust", dust_density, 0.0, 0.05, 0.001, env_f32("dust_density"))}
                    {prop_slider(__scope, "Dust Asymmetry", dust_asymmetry, 0.0, 1.0, 0.01, env_f32("dust_asymmetry"))}
                    {prop_slider(__scope, "Height Fog", height_fog_density, 0.0, 0.5, 0.01, env_f32("height_fog_density"))}
                    {prop_slider(__scope, "Base Height", fog_base_height, -50.0, 100.0, 1.0, env_f32("fog_base_height"))}
                    {prop_slider(__scope, "Height Falloff", fog_height_falloff, 0.01, 1.0, 0.01, env_f32("fog_height_falloff"))}
                    {prop_slider(__scope, "Distance Fog", distance_fog_density, 0.0, 0.1, 0.001, env_f32("distance_fog_density"))}
                    {prop_slider(__scope, "Far Distance", vol_far, 50.0, 1000.0, 10.0, env_f32("vol_far"))}
                }
            }

            // ── Clouds section ──────────────────────────────────────────
            {prop_section_header(__scope, "Clouds", cloud_collapsed, None)}
            if !cloud_collapsed.get() {
                div {
                    style: "padding:6px 12px;display:flex;flex-direction:column;gap:4px;",
                    {prop_checkbox(__scope, "Enabled", clouds_enabled, env_bool("clouds_enabled"))}
                    {prop_slider(__scope, "Min Altitude", cloud_altitude_min, 0.0, 5000.0, 50.0, env_f32("cloud_altitude_min"))}
                    {prop_slider(__scope, "Max Altitude", cloud_altitude_max, 0.0, 10000.0, 100.0, env_f32("cloud_altitude_max"))}
                    {prop_slider(__scope, "Coverage", cloud_coverage, 0.0, 1.0, 0.01, env_f32("cloud_coverage"))}
                    {prop_slider(__scope, "Density", cloud_density_scale, 0.0, 5.0, 0.1, env_f32("cloud_density_scale"))}
                    {prop_slider(__scope, "Wind Speed", cloud_wind_speed, 0.0, 20.0, 0.5, env_f32("cloud_wind_speed"))}
                    {prop_slider(__scope, "Wind Dir", cloud_wind_dir, 0.0, 360.0, 1.0, env_f32("cloud_wind_dir"))}
                }
            }
        }
    }
}

/// Sun direction widget — azimuth slider with compass labels + elevation slider.
fn sun_direction_widget(
    __scope: &mut rinch::core::dom::RenderScope,
    azimuth: Signal<f32>,
    elevation: Signal<f32>,
    cmd_tx: CmdSignal,
) -> rinch::core::dom::NodeHandle {
    // Compass label that updates reactively.
    let compass_label = move || {
        let az = azimuth.get();
        let dir = match az as u32 {
            0..=22 | 338..=360 => "N",
            23..=67 => "NE",
            68..=112 => "E",
            113..=157 => "SE",
            158..=202 => "S",
            203..=247 => "SW",
            248..=292 => "W",
            293..=337 => "NW",
            _ => "",
        };
        format!("{:.0}\u{00B0} {dir}", az)
    };

    // Bridge f32 signals to f64 for rinch Slider.
    let az_f64 = Signal::new(azimuth.get() as f64);
    let el_f64 = Signal::new(elevation.get() as f64);

    rsx! {
        div {
            style: "display:flex;flex-direction:column;gap:4px;",

            // Azimuth — 0°–360° with compass readout
            div {
                style: "display:flex;align-items:center;gap:6px;min-height:22px;",
                div {
                    style: "width:72px;flex-shrink:0;font-size:11px;color:#999;",
                    {"Azimuth"}
                }
                div {
                    style: "flex:1;min-width:0;",
                    Slider {
                        min: 0.0,
                        max: 360.0,
                        step: 1.0,
                        size: "sm",
                        color: "#4fc3f7",
                        value_signal: az_f64,
                        onchange: move |v: f64| {
                            let f = v as f32;
                            azimuth.set(f);
                            az_f64.set(v);
                            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::UpdateEnvironment {
                                field: "sun_azimuth".into(),
                                value: f.to_string(),
                            });
                        },
                    }
                }
                div {
                    style: "width:48px;text-align:right;font-size:10px;color:#777;\
                            font-family:monospace;flex-shrink:0;",
                    {move || compass_label()}
                }
            }

            // Elevation — -10° to 90°
            div {
                style: "display:flex;align-items:center;gap:6px;min-height:22px;",
                div {
                    style: "width:72px;flex-shrink:0;font-size:11px;color:#999;",
                    {"Elevation"}
                }
                div {
                    style: "flex:1;min-width:0;",
                    Slider {
                        min: -10.0,
                        max: 90.0,
                        step: 1.0,
                        size: "sm",
                        color: "#4fc3f7",
                        value_signal: el_f64,
                        onchange: move |v: f64| {
                            let f = v as f32;
                            elevation.set(f);
                            el_f64.set(v);
                            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::UpdateEnvironment {
                                field: "sun_elevation".into(),
                                value: f.to_string(),
                            });
                        },
                    }
                }
                div {
                    style: "width:48px;text-align:right;font-size:10px;color:#777;\
                            font-family:monospace;flex-shrink:0;",
                    {move || format!("{:.0}\u{00B0}", elevation.get())}
                }
            }
        }
    }
}
