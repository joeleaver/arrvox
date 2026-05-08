//! Environment panel — sky, lighting, shadows, AO, and tone mapping settings.
//!
//! Reactivity model: every field is a `Memo<T>` derived from
//! `store.environment`. There is **no** remount-via-key wrapper — when
//! the engine pushes a new env, only the Memos whose values actually
//! changed invalidate, and only those fields' DOM nodes update. Edits
//! fire `EngineCommand::UpdateEnvironment`; the engine accepts, the
//! diff-suppressed `build_state_update` echoes the new env back, and
//! the source-of-truth Memo refires for the one changed field.

use std::rc::Rc;

use rinch::prelude::*;

use rkp_engine::environment::{
    cloud_quality_values, shadow_quality_values, CloudQualityPreset, ShadowQualityPreset,
};

use crate::CommandSender;
use crate::ui::store::EditorStore;
use super::prop_controls::*;

type CmdSignal = Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>;

#[component]
pub fn EnvironmentPanel() -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd_tx: CmdSignal = Signal::new(use_context::<CommandSender>().0);

    // Per-field Memos. Each only invalidates downstream Effects when its
    // own field's value changes (PartialEq short-circuit), so a coverage
    // tweak doesn't churn ten unrelated sliders.
    let ambient = Memo::new(move || store.environment.get().ambient_intensity);
    let scene_elevation = Memo::new(move || store.environment.get().scene_elevation);
    let ground_albedo = Memo::new(move || {
        let c = store.environment.get().ground_albedo;
        [c[0], c[1], c[2], 1.0]
    });

    let sky_top_override_on = Memo::new(move || store.environment.get().sky_color_top_override.is_some());
    let sky_top_color = Memo::new(move || {
        let c = store.environment.get().sky_color_top_override.unwrap_or([0.4, 0.6, 1.0]);
        [c[0], c[1], c[2], 1.0]
    });
    let sky_horizon_override_on = Memo::new(move || store.environment.get().sky_color_horizon_override.is_some());
    let sky_horizon_color = Memo::new(move || {
        let c = store.environment.get().sky_color_horizon_override.unwrap_or([0.8, 0.85, 0.9]);
        [c[0], c[1], c[2], 1.0]
    });

    let sun_azimuth = Memo::new(move || store.environment.get().sun_azimuth);
    let sun_elevation = Memo::new(move || store.environment.get().sun_elevation);
    let sun_color_override_on = Memo::new(move || store.environment.get().sun_color_override.is_some());
    let sun_color_override = Memo::new(move || {
        let c = store.environment.get().sun_color_override.unwrap_or([1.0, 0.95, 0.9]);
        [c[0], c[1], c[2], 1.0]
    });
    let sun_intensity = Memo::new(move || store.environment.get().sun_intensity);

    let shadow_steps = Memo::new(move || store.environment.get().shadow_steps as f32);
    let shadow_csm_max_distance = Memo::new(move || store.environment.get().shadow_csm_max_distance);
    let shadow_csm_lambda = Memo::new(move || store.environment.get().shadow_csm_lambda);
    let shadow_csm_depth_bias = Memo::new(move || store.environment.get().shadow_csm_depth_bias);
    let shadow_csm_threshold_falloff = Memo::new(move || store.environment.get().shadow_csm_threshold_falloff);
    let shadow_csm_sharp_distance = Memo::new(move || store.environment.get().shadow_csm_sharp_distance);
    let ao_radius = Memo::new(move || store.environment.get().ao_radius);
    let ao_steps = Memo::new(move || store.environment.get().ao_steps as f32);

    let exposure = Memo::new(move || store.environment.get().exposure);

    let bloom_threshold = Memo::new(move || store.environment.get().bloom_threshold);
    let bloom_knee = Memo::new(move || store.environment.get().bloom_knee);
    let bloom_intensity = Memo::new(move || store.environment.get().bloom_intensity);

    let god_ray_density = Memo::new(move || store.environment.get().god_ray_density);
    let god_ray_exposure = Memo::new(move || store.environment.get().god_ray_exposure);
    let god_ray_decay = Memo::new(move || store.environment.get().god_ray_decay);

    let height_fog_density = Memo::new(move || store.environment.get().height_fog_density);
    let fog_base_height = Memo::new(move || store.environment.get().fog_base_height);
    let fog_height_falloff = Memo::new(move || store.environment.get().fog_height_falloff);
    let fog_color = Memo::new(move || {
        let c = store.environment.get().fog_color;
        [c[0], c[1], c[2], 1.0]
    });
    let vol_far = Memo::new(move || store.environment.get().vol_far);

    let clouds_enabled = Memo::new(move || store.environment.get().clouds_enabled);
    let attenuate_sun_by_clouds = Memo::new(move || store.environment.get().attenuate_sun_by_clouds);
    let cloud_altitude_min = Memo::new(move || store.environment.get().cloud_altitude_min);
    let cloud_altitude_max = Memo::new(move || store.environment.get().cloud_altitude_max);
    let cloud_coverage = Memo::new(move || store.environment.get().cloud_coverage);
    let cloud_density_scale = Memo::new(move || store.environment.get().cloud_density_scale);
    let cloud_wind_speed = Memo::new(move || store.environment.get().cloud_wind_speed);
    let cloud_wind_dir = Memo::new(move || store.environment.get().cloud_wind_dir);

    let cloud_slab_steps = Memo::new(move || store.environment.get().cloud_slab_steps as f32);
    let cloud_shadow_steps = Memo::new(move || store.environment.get().cloud_shadow_steps as f32);
    let cloud_detail_octaves = Memo::new(move || store.environment.get().cloud_detail_octaves as f32);
    let cloud_ms_octaves = Memo::new(move || store.environment.get().cloud_ms_octaves as f32);
    let cloud_taa_alpha = Memo::new(move || store.environment.get().cloud_taa_alpha);

    // Section collapsed-state stays as local Signals — purely UI state,
    // not part of the engine env.
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

    // Cloud quality preset application — fires one command per field; the
    // engine echoes the new values back via the env Memos above.
    let apply_preset: Rc<dyn Fn(CloudQualityPreset)> = {
        Rc::new(move |preset: CloudQualityPreset| {
            let (slab, shadow, detail, ms, alpha) = cloud_quality_values(preset);
            let tx = cmd_tx.get();
            let send = |field: &str, v: String| {
                let _ = tx.send(rkp_engine::EngineCommand::UpdateEnvironment {
                    field: field.into(),
                    value: v,
                });
            };
            send("cloud_slab_steps", (slab as f32).to_string());
            send("cloud_shadow_steps", (shadow as f32).to_string());
            send("cloud_detail_octaves", (detail as f32).to_string());
            send("cloud_ms_octaves", (ms as f32).to_string());
            send("cloud_taa_alpha", alpha.to_string());
        })
    };

    // Shadow Quality preset — bundles the per-cascade falloff knob
    // (and λ, pinned at 0.95). Sets `shadow_csm_threshold_falloff`
    // and `shadow_csm_lambda` on apply; the active row is derived
    // from those two reading them back through env Memos.
    let shadow_apply_preset: Rc<dyn Fn(ShadowQualityPreset)> = {
        Rc::new(move |preset: ShadowQualityPreset| {
            let (falloff, lambda) = shadow_quality_values(preset);
            let tx = cmd_tx.get();
            let send = |field: &str, v: String| {
                let _ = tx.send(rkp_engine::EngineCommand::UpdateEnvironment {
                    field: field.into(),
                    value: v,
                });
            };
            send("shadow_csm_threshold_falloff", falloff.to_string());
            send("shadow_csm_lambda", lambda.to_string());
        })
    };
    let shadow_active_preset = Memo::new(move || -> Option<ShadowQualityPreset> {
        let f = shadow_csm_threshold_falloff.get();
        let l = shadow_csm_lambda.get();
        for p in [
            ShadowQualityPreset::Low,
            ShadowQualityPreset::Medium,
            ShadowQualityPreset::High,
        ] {
            let (pf, pl) = shadow_quality_values(p);
            if (f - pf).abs() < 0.01 && (l - pl).abs() < 0.001 {
                return Some(p);
            }
        }
        None
    });

    // Active preset — derived Memo so preset buttons re-style only when
    // the preset selection actually changes.
    let active_preset = Memo::new(move || -> Option<CloudQualityPreset> {
        let s = cloud_slab_steps.get() as u32;
        let sh = cloud_shadow_steps.get() as u32;
        let d = cloud_detail_octaves.get() as u32;
        let m = cloud_ms_octaves.get() as u32;
        let a = cloud_taa_alpha.get();
        for p in [
            CloudQualityPreset::Low,
            CloudQualityPreset::Medium,
            CloudQualityPreset::High,
            CloudQualityPreset::Ultra,
        ] {
            let (ps, psh, pd, pm, pa) = cloud_quality_values(p);
            if s == ps && sh == psh && d == pd && m == pm && (a - pa).abs() < 0.001 {
                return Some(p);
            }
        }
        None
    });

    rsx! {
        div {
            style: "display:flex;flex-direction:column;height:100%;overflow-y:auto;\
                    color:#ccc;font-size:12px;",

            // ── Atmosphere section ────────────────────────────────────
            {prop_section_header(__scope, "Atmosphere", sky_collapsed, None)}
            if !sky_collapsed.get() {
                div {
                    style: "padding:6px 12px;display:flex;flex-direction:column;gap:4px;",
                    {prop_slider_memo(__scope, "Ambient", ambient, 0.0, 5.0, 0.1, env_f32("ambient_intensity"))}
                    {prop_slider_memo(__scope, "Scene Elevation (m)", scene_elevation, 0.0, 9000.0, 10.0, env_f32("scene_elevation"))}
                    {prop_color(__scope, "Ground Color", ground_albedo, env_color3("ground_albedo"))}

                    // Override: sky top color. Toggling off sends the
                    // disable command directly; toggling on sends the
                    // current display color (the engine creates the
                    // Some(...) on first non-empty value).
                    {prop_checkbox_memo(__scope, "Override Sky Top", sky_top_override_on, {
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
                    {prop_checkbox_memo(__scope, "Override Sky Horizon", sky_horizon_override_on, {
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
                    {prop_slider_memo(__scope, "Intensity (lux)", sun_intensity, 0.0, 200000.0, 1000.0, env_f32("sun_intensity"))}

                    // Override: sun color (bypasses atmosphere extinction)
                    {prop_checkbox_memo(__scope, "Override Sun Color", sun_color_override_on, {
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
                    {prop_slider_memo(__scope, "Shadow Steps", shadow_steps, 4.0, 64.0, 4.0, env_f32("shadow_steps"))}
                    // Sharp Distance: how far the highest-detail shadow
                    // cascade extends. Cascade 0 covers `[near, sharp]`,
                    // remaining cascades distribute over `[sharp, max]`.
                    // Default 2 m is fine for prop-scale assets; raise
                    // for terrain / outdoor vistas where you want
                    // pixel-perfect shadows further out.
                    {prop_slider_memo(__scope, "Sharp Distance (m)", shadow_csm_sharp_distance, 0.5, 50.0, 0.5, env_f32("shadow_csm_sharp_distance"))}
                    // Max Distance: the far cap. Anything beyond falls
                    // back to fully-lit. Lower = sharper everywhere
                    // (cascades pack tighter); higher = shadows reach
                    // further but each cascade covers more area.
                    {prop_slider_memo(__scope, "Max Distance (m)", shadow_csm_max_distance, 10.0, 500.0, 5.0, env_f32("shadow_csm_max_distance"))}
                    // Quality preset — bundles the per-cascade detail
                    // falloff. Low = far cascades drop to coarse
                    // geometry quickly (fastest); High = far cascades
                    // stay close to cascade-0 quality (most expensive).
                    div {
                        style: "font-size:11px;color:#888;margin-top:8px;",
                        {"Shadow Quality"}
                    }
                    div {
                        style: "display:flex;gap:4px;",
                        {preset_button(__scope, "Low", ShadowQualityPreset::Low, shadow_apply_preset.clone(), shadow_active_preset)}
                        {preset_button(__scope, "Medium", ShadowQualityPreset::Medium, shadow_apply_preset.clone(), shadow_active_preset)}
                        {preset_button(__scope, "High", ShadowQualityPreset::High, shadow_apply_preset.clone(), shadow_active_preset)}
                    }
                    {prop_slider_memo(__scope, "Shadow Depth Bias", shadow_csm_depth_bias, 0.0, 0.01, 0.0001, env_f32("shadow_csm_depth_bias"))}
                    {prop_slider_memo(__scope, "AO Radius", ao_radius, 0.01, 1.0, 0.01, env_f32("ao_radius"))}
                    {prop_slider_memo(__scope, "AO Steps", ao_steps, 1.0, 16.0, 1.0, env_f32("ao_steps"))}
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
                        exposure,
                        0.000001, 0.01, 0.000001,
                        env_f32("exposure"),
                    )}
                }
            }

            // ── Bloom section ────────────────────────────────────────────
            {prop_section_header(__scope, "Bloom", bloom_collapsed, None)}
            if !bloom_collapsed.get() {
                div {
                    style: "padding:6px 12px;display:flex;flex-direction:column;gap:4px;",
                    {prop_slider_memo(__scope, "Threshold", bloom_threshold, 0.0, 5.0, 0.1, env_f32("bloom_threshold"))}
                    {prop_slider_memo(__scope, "Knee", bloom_knee, 0.0, 1.0, 0.01, env_f32("bloom_knee"))}
                    {prop_slider_memo(__scope, "Intensity", bloom_intensity, 0.0, 2.0, 0.01, env_f32("bloom_intensity"))}
                }
            }

            // ── God Rays section ─────────────────────────────────────────
            {prop_section_header(__scope, "God Rays", god_ray_collapsed, None)}
            if !god_ray_collapsed.get() {
                div {
                    style: "padding:6px 12px;display:flex;flex-direction:column;gap:4px;",
                    {prop_slider_memo(__scope, "Density", god_ray_density, 0.0, 1.0, 0.05, env_f32("god_ray_density"))}
                    {prop_slider_memo(__scope, "Exposure", god_ray_exposure, 0.0, 1.0, 0.01, env_f32("god_ray_exposure"))}
                    {prop_slider_memo(__scope, "Decay", god_ray_decay, 0.85, 1.0, 0.005, env_f32("god_ray_decay"))}
                }
            }

            // ── Fog section ─────────────────────────────────────────────
            {prop_section_header(__scope, "Fog", fog_collapsed, None)}
            if !fog_collapsed.get() {
                div {
                    style: "padding:6px 12px;display:flex;flex-direction:column;gap:4px;",
                    {prop_color(__scope, "Fog Color", fog_color, env_color3("fog_color"))}
                    {prop_slider_memo(__scope, "Height Fog", height_fog_density, 0.0, 0.5, 0.01, env_f32("height_fog_density"))}
                    {prop_slider_memo(__scope, "Base Height", fog_base_height, -50.0, 100.0, 1.0, env_f32("fog_base_height"))}
                    {prop_slider_memo(__scope, "Height Falloff", fog_height_falloff, 0.01, 1.0, 0.01, env_f32("fog_height_falloff"))}
                    {prop_slider_memo(__scope, "Far Distance", vol_far, 50.0, 1000.0, 10.0, env_f32("vol_far"))}
                }
            }

            // ── Clouds section ──────────────────────────────────────────
            {prop_section_header(__scope, "Clouds", cloud_collapsed, None)}
            if !cloud_collapsed.get() {
                div {
                    style: "padding:6px 12px;display:flex;flex-direction:column;gap:4px;",
                    {prop_checkbox_memo(__scope, "Enabled", clouds_enabled, env_bool("clouds_enabled"))}
                    {prop_checkbox_memo(__scope, "Attenuate Sun Intensity", attenuate_sun_by_clouds, env_bool("attenuate_sun_by_clouds"))}
                    {prop_slider_memo(__scope, "Min Altitude", cloud_altitude_min, 0.0, 5000.0, 50.0, env_f32("cloud_altitude_min"))}
                    {prop_slider_memo(__scope, "Max Altitude", cloud_altitude_max, 0.0, 10000.0, 100.0, env_f32("cloud_altitude_max"))}
                    {prop_slider_memo(__scope, "Coverage", cloud_coverage, 0.0, 1.0, 0.01, env_f32("cloud_coverage"))}
                    {prop_slider_memo(__scope, "Density", cloud_density_scale, 0.0, 1.0, 0.01, env_f32("cloud_density_scale"))}
                    {prop_slider_memo(__scope, "Wind Speed", cloud_wind_speed, 0.0, 20.0, 0.5, env_f32("cloud_wind_speed"))}
                    {prop_slider_memo(__scope, "Wind Dir", cloud_wind_dir, 0.0, 360.0, 1.0, env_f32("cloud_wind_dir"))}

                    // Quality presets — bundle the render-cost knobs. Individual
                    // sliders below let users override specific values; the row
                    // highlights whichever preset the current settings match.
                    div {
                        style: "font-size:11px;color:#888;margin-top:8px;",
                        {"Quality"}
                    }
                    div {
                        style: "display:flex;gap:4px;",
                        {preset_button(__scope, "Low", CloudQualityPreset::Low, apply_preset.clone(), active_preset)}
                        {preset_button(__scope, "Medium", CloudQualityPreset::Medium, apply_preset.clone(), active_preset)}
                        {preset_button(__scope, "High", CloudQualityPreset::High, apply_preset.clone(), active_preset)}
                        {preset_button(__scope, "Ultra", CloudQualityPreset::Ultra, apply_preset.clone(), active_preset)}
                    }
                    {prop_slider_memo(__scope, "Slab Samples", cloud_slab_steps, 8.0, 64.0, 1.0, env_f32("cloud_slab_steps"))}
                    {prop_slider_memo(__scope, "Shadow Samples", cloud_shadow_steps, 1.0, 6.0, 1.0, env_f32("cloud_shadow_steps"))}
                    {prop_slider_memo(__scope, "Detail Octaves", cloud_detail_octaves, 1.0, 5.0, 1.0, env_f32("cloud_detail_octaves"))}
                    {prop_slider_memo(__scope, "Multi-scatter Octaves", cloud_ms_octaves, 1.0, 4.0, 1.0, env_f32("cloud_ms_octaves"))}
                    {prop_slider_memo(__scope, "TAA Weight", cloud_taa_alpha, 0.05, 0.7, 0.01, env_f32("cloud_taa_alpha"))}
                }
            }
        }
    }
}

/// One entry in the cloud quality preset row. Highlights itself when the
/// active preset Memo matches; clicking fires `apply_preset`.
fn preset_button<P: Copy + PartialEq + 'static>(
    __scope: &mut rinch::core::dom::RenderScope,
    label: &str,
    preset: P,
    apply: Rc<dyn Fn(P)>,
    active: Memo<Option<P>>,
) -> rinch::core::dom::NodeHandle {
    const BASE: &str = "flex:1;padding:4px 8px;font-size:11px;border-radius:3px;\
                        cursor:pointer;border:1px solid #333;";
    let label = label.to_string();
    rsx! {
        div {
            style: {move || {
                if active.get() == Some(preset) {
                    format!("{BASE}background:#3a6a9a;color:#fff;border-color:#4a7aaa;")
                } else {
                    format!("{BASE}background:#222;color:#bbb;")
                }
            }},
            onclick: move || apply(preset),
            {label}
        }
    }
}

/// Sun direction widget — azimuth slider with compass labels + elevation slider.
///
/// Both fields read reactively from `azimuth`/`elevation` Memos backed
/// by `store.environment`. The local Signals (`az_f64`, `el_f64`) only
/// exist to bridge the rinch `Slider` component, which still wants a
/// `Signal<f64>` source — they're updated on user drag, but the
/// authoritative value the rest of the panel sees comes from the store.
fn sun_direction_widget(
    __scope: &mut rinch::core::dom::RenderScope,
    azimuth: Memo<f32>,
    elevation: Memo<f32>,
    cmd_tx: CmdSignal,
) -> rinch::core::dom::NodeHandle {
    // Compass label that updates reactively against the store-backed Memo.
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

    // Bridge f32 Memos to f64 Signals for rinch Slider. We seed once at
    // mount and update on drag; external store updates won't push back
    // into these (Slider still reads from `value_signal`), but a
    // store-only edit through e.g. MCP would still be visible via the
    // compass label and the elevation readout. Replacing rinch's Slider
    // with a Memo-aware variant is a future cleanup.
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
