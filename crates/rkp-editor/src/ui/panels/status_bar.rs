//! Status bar — displays engine state + play/stop controls.

use rinch::prelude::*;
use rinch_tabler_icons::{TablerIcon, TablerIconStyle, render_tabler_icon};

use crate::CommandSender;
use crate::ui::store::EditorStore;

#[component]
pub fn StatusBar() -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd = use_context::<CommandSender>();

    rsx! {
        div {
            style: "height:28px;display:flex;align-items:center;padding:0 12px;\
                    background:#252526;color:#858585;font-size:11px;flex-shrink:0;\
                    gap:12px;border-top:1px solid #3c3c3c;",

            // Play/Stop button
            div {
                style: {move || {
                    if store.play_mode.get() {
                        "display:flex;align-items:center;gap:4px;padding:2px 10px;\
                         background:#5a2d2d;border:1px solid #8b3a3a;border-radius:3px;\
                         cursor:pointer;color:#ef5350;font-size:11px;font-weight:600;"
                    } else {
                        "display:flex;align-items:center;gap:4px;padding:2px 10px;\
                         background:#2d5a2d;border:1px solid #3c7c3c;border-radius:3px;\
                         cursor:pointer;color:#4caf50;font-size:11px;font-weight:600;"
                    }
                }},
                onclick: {
                    let cmd = cmd.clone();
                    move || {
                        if store.play_mode.get() {
                            let _ = cmd.0.send(rkp_engine::EngineCommand::PlayStop);
                        } else {
                            let _ = cmd.0.send(rkp_engine::EngineCommand::PlayStart);
                        }
                    }
                },
                if store.play_mode.get() {
                    span {
                        style: "width:12px;height:12px;display:inline-flex;\
                                align-items:center;justify-content:center;",
                        {render_tabler_icon(__scope, TablerIcon::PlayerStop, TablerIconStyle::Filled)}
                    }
                    {"Stop"}
                }
                if !store.play_mode.get() {
                    span {
                        style: "width:12px;height:12px;display:inline-flex;\
                                align-items:center;justify-content:center;",
                        {render_tabler_icon(__scope, TablerIcon::PlayerPlay, TablerIconStyle::Filled)}
                    }
                    {"Play"}
                }
            }

            // Playing indicator
            if store.play_mode.get() {
                div {
                    style: "color:#4caf50;font-weight:600;font-size:11px;",
                    {"PLAYING"}
                }
            }

            // Spacer
            div { style: "flex:1;" }

            // Stats
            span { {|| format!("{} objects", store.gpu_object_count.get())} }
            span { {|| format!("{:.0} fps", store.fps.get())} }
            span { {|| format!("{:?}", store.gizmo_mode.get())} }
        }
    }
}
