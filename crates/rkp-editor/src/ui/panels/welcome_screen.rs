//! Welcome screen — shown when no project is loaded.

use rinch::prelude::*;
use rinch_tabler_icons::{TablerIcon, TablerIconStyle, render_tabler_icon};

use crate::CommandSender;
use crate::ui::store::EditorStore;
use rkp_engine::recent_projects::RecentProject;

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

                // Recent projects list
                if !store.recent_projects.get().is_empty() {
                    div {
                        style: "width:100%;margin-top:8px;",
                        div {
                            style: "font-size:11px;font-weight:600;color:#888;text-transform:uppercase;\
                                    letter-spacing:0.5px;margin-bottom:8px;",
                            {"Recent Projects"}
                        }
                        for project in store.recent_projects.get() {
                            RecentProjectRow {
                                key: project.path.clone(),
                                project: project.clone(),
                            }
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn RecentProjectRow(project: RecentProject) -> NodeHandle {
    let cmd = use_context::<CommandSender>();
    let path = project.path.clone();
    let name = project.name.clone();
    let exists = std::path::Path::new(&path).exists();

    rsx! {
        div {
            style: {|| {
                if exists {
                    "display:flex;align-items:center;gap:8px;padding:8px 12px;\
                     background:#2a2a2a;border-radius:4px;margin-bottom:4px;\
                     cursor:pointer;color:#ccc;font-size:12px;"
                } else {
                    "display:flex;align-items:center;gap:8px;padding:8px 12px;\
                     background:#2a2a2a;border-radius:4px;margin-bottom:4px;\
                     cursor:default;color:#555;font-size:12px;opacity:0.5;"
                }
            }},
            onclick: {
                let path = path.clone();
                let cmd = cmd.clone();
                move || {
                    if exists {
                        let _ = cmd.0.send(rkp_engine::EngineCommand::OpenProject {
                            path: path.clone(),
                        });
                    }
                }
            },
            // Status dot
            div {
                style: {|| format!(
                    "width:6px;height:6px;border-radius:50%;flex-shrink:0;background:{};",
                    if exists { "#4caf50" } else { "#f44336" }
                )},
            }
            // Name + path
            div {
                style: "flex:1;min-width:0;",
                div { style: "font-weight:500;", {name} }
                div {
                    style: "font-size:10px;color:#666;overflow:hidden;\
                            text-overflow:ellipsis;white-space:nowrap;",
                    {path}
                }
            }
        }
    }
}
