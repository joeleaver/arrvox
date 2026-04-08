//! Welcome screen — shown when no project is loaded.

use rinch::prelude::*;
use rinch_tabler_icons::{TablerIcon, TablerIconStyle, render_tabler_icon};

use crate::CommandSender;
use crate::ui::store::EditorStore;

#[component]
pub fn WelcomeScreen() -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd = use_context::<CommandSender>();

    rsx! {
        div {
            style: "position:absolute;inset:0;z-index:300;\
                    display:flex;align-items:center;justify-content:center;\
                    background:#1e1e1e;",

            div {
                style: "display:flex;flex-direction:column;align-items:center;\
                        gap:24px;max-width:500px;width:100%;",

                // Title
                div { style: "font-size:28px;font-weight:700;color:#ddd;letter-spacing:-0.5px;",
                    {"RKIPatch"}
                }
                div { style: "font-size:13px;color:#888;margin-top:-16px;",
                    {"Gaussian Splat Graphics Engine"}
                }

                // Action buttons
                div {
                    style: "display:flex;gap:12px;margin-top:16px;",

                    // New Project
                    div {
                        style: "display:flex;align-items:center;gap:8px;padding:10px 20px;\
                                background:#2d2d2d;border:1px solid #3c3c3c;border-radius:6px;\
                                cursor:pointer;color:#ccc;font-size:13px;",
                        onclick: {
                            let cmd = cmd.clone();
                            move || {
                                if let Some(path) = rfd::FileDialog::new()
                                    .set_title("New Project")
                                    .add_filter("RKIPatch Project", &["rkproject"])
                                    .save_file()
                                {
                                    let _ = cmd.0.send(rkp_engine::EngineCommand::NewProject {
                                        path: path.to_string_lossy().into_owned(),
                                    });
                                }
                            }
                        },
                        span {
                            style: "width:16px;height:16px;display:inline-flex;\
                                    align-items:center;justify-content:center;color:#999;",
                            {render_tabler_icon(__scope, TablerIcon::FolderPlus, TablerIconStyle::Outline)}
                        }
                        {"New Project"}
                    }

                    // Open Project
                    div {
                        style: "display:flex;align-items:center;gap:8px;padding:10px 20px;\
                                background:#2d2d2d;border:1px solid #3c3c3c;border-radius:6px;\
                                cursor:pointer;color:#ccc;font-size:13px;",
                        onclick: {
                            let cmd = cmd.clone();
                            move || {
                                if let Some(path) = rfd::FileDialog::new()
                                    .set_title("Open Project")
                                    .add_filter("RKIPatch Project", &["rkproject"])
                                    .pick_file()
                                {
                                    let _ = cmd.0.send(rkp_engine::EngineCommand::OpenProject {
                                        path: path.to_string_lossy().into_owned(),
                                    });
                                }
                            }
                        },
                        span {
                            style: "width:16px;height:16px;display:inline-flex;\
                                    align-items:center;justify-content:center;color:#999;",
                            {render_tabler_icon(__scope, TablerIcon::FolderOpen, TablerIconStyle::Outline)}
                        }
                        {"Open Project"}
                    }
                }
            }
        }
    }
}
