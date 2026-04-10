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

    // Create signals from current environment state.
    // These are initialized once — the engine pushes updates via store.environment.
    let env = store.environment.get();

    let sky_top = Signal::new([env.sky_color_top[0], env.sky_color_top[1], env.sky_color_top[2], 1.0]);
    let sky_horizon = Signal::new([env.sky_color_horizon[0], env.sky_color_horizon[1], env.sky_color_horizon[2], 1.0]);
    let ambient = Signal::new(env.ambient_intensity);

    let sun_dir_x = Signal::new(env.sun_direction[0]);
    let sun_dir_y = Signal::new(env.sun_direction[1]);
    let sun_dir_z = Signal::new(env.sun_direction[2]);
    let sun_color = Signal::new([env.sun_color[0], env.sun_color[1], env.sun_color[2], 1.0]);
    let sun_intensity = Signal::new(env.sun_intensity);

    let shadow_steps = Signal::new(env.shadow_steps as f32);
    let ao_radius = Signal::new(env.ao_radius);
    let ao_steps = Signal::new(env.ao_steps as f32);

    let exposure = Signal::new(env.exposure);

    // Collapsed state for each section.
    let sky_collapsed = Signal::new(false);
    let light_collapsed = Signal::new(false);
    let shadow_collapsed = Signal::new(false);
    let tone_collapsed = Signal::new(false);

    // Helper to send environment updates.
    let env_f32 = move |field: &'static str| -> Rc<dyn Fn(f32)> {
        Rc::new(move |v: f32| {
            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::UpdateEnvironment {
                field: field.into(),
                value: v.to_string(),
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

            // ── Sky section ──────────────────────────────────────────
            {prop_section_header(__scope, "Sky", sky_collapsed, None)}
            if !sky_collapsed.get() {
                div {
                    style: "padding:6px 12px;display:flex;flex-direction:column;gap:4px;",
                    {prop_color(__scope, "Top", sky_top, env_color3("sky_color_top"))}
                    {prop_color(__scope, "Horizon", sky_horizon, env_color3("sky_color_horizon"))}
                    {prop_slider(__scope, "Ambient", ambient, 0.0, 2.0, 0.01, env_f32("ambient_intensity"))}
                }
            }

            // ── Sun / Lighting section ───────────────────────────────
            {prop_section_header(__scope, "Sun", light_collapsed, None)}
            if !light_collapsed.get() {
                div {
                    style: "padding:6px 12px;display:flex;flex-direction:column;gap:4px;",
                    {prop_color(__scope, "Color", sun_color, env_color3("sun_color"))}
                    {prop_slider(__scope, "Intensity", sun_intensity, 0.0, 10.0, 0.1, env_f32("sun_intensity"))}
                    {prop_slider(__scope, "Dir X", sun_dir_x, -1.0, 1.0, 0.01, {
                        let cmd_tx = cmd_tx;
                        Rc::new(move |v: f32| {
                            // Read current direction, update X, send as array.
                            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::UpdateEnvironment {
                                field: "sun_direction".into(),
                                value: format!("[{},{},{}]", v, sun_dir_y.get(), sun_dir_z.get()),
                            });
                        })
                    })}
                    {prop_slider(__scope, "Dir Y", sun_dir_y, -1.0, 1.0, 0.01, {
                        let cmd_tx = cmd_tx;
                        Rc::new(move |v: f32| {
                            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::UpdateEnvironment {
                                field: "sun_direction".into(),
                                value: format!("[{},{},{}]", sun_dir_x.get(), v, sun_dir_z.get()),
                            });
                        })
                    })}
                    {prop_slider(__scope, "Dir Z", sun_dir_z, -1.0, 1.0, 0.01, {
                        let cmd_tx = cmd_tx;
                        Rc::new(move |v: f32| {
                            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::UpdateEnvironment {
                                field: "sun_direction".into(),
                                value: format!("[{},{},{}]", sun_dir_x.get(), sun_dir_y.get(), v),
                            });
                        })
                    })}
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
                    {prop_slider(__scope, "Exposure", exposure, 0.1, 10.0, 0.1, env_f32("exposure"))}
                }
            }
        }
    }
}
