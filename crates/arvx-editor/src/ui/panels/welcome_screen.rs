//! Welcome screen — shown when no project is loaded.

use rinch::prelude::*;
use rinch_tabler_icons::{TablerIcon, TablerIconStyle, render_tabler_icon};

use crate::CommandSender;
use crate::ui::store::EditorStore;
use arvx_engine::recent_projects::RecentProject;
use arvx_engine::snapshot::{ProjectLoadPhase, ProjectLoadingStatus};

/// Derive a display name from a project file path. `New Project`
/// chooses the save-file's stem; `Open Project` and recents already
/// have a known name.
fn project_name_from_path(path: &std::path::Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_owned())
        .unwrap_or_else(|| "project".into())
}

#[component]
pub fn WelcomeScreen() -> NodeHandle {
    let store = use_context::<EditorStore>();

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
                    {"Arrvox"}
                }
                div { style: "font-size:13px;color:#888;margin-top:-16px;",
                    {"A Rust & Rinch Voxel Game Engine. For Pirates."}
                }

                // Body — either the loading panel (while opening /
                // creating a project) or the idle layout (New / Open
                // / Recents). Driven by `store.project_loading`; the
                // engine streams phase updates via `publish_phase`.
                if store.project_loading.get().is_some() {
                    LoadingPanel {}
                } else {
                    IdleLayout {}
                }
            }
        }
    }
}

#[component]
fn IdleLayout() -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd = use_context::<CommandSender>();

    rsx! {
        div {
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
                                .add_filter("Arrvox Project", &["arvxproject"])
                                .save_file()
                            {
                                // Flip the loading panel on immediately
                                // so the user sees feedback before the
                                // first engine phase ping lands.
                                let name = project_name_from_path(&path);
                                store.project_loading.send(Some(ProjectLoadingStatus {
                                    project_name: name,
                                    phase: ProjectLoadPhase::Finalizing,
                                    detail: Some("Creating project…".into()),
                                }));
                                let _ = cmd.0.send(arvx_engine::EngineCommand::NewProject {
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
                                .add_filter("Arrvox Project", &["arvxproject"])
                                .pick_file()
                            {
                                let name = project_name_from_path(&path);
                                store.project_loading.send(Some(ProjectLoadingStatus {
                                    project_name: name,
                                    phase: ProjectLoadPhase::ScaffoldGameplay,
                                    detail: None,
                                }));
                                let _ = cmd.0.send(arvx_engine::EngineCommand::OpenProject {
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

#[component]
fn LoadingPanel() -> NodeHandle {
    let store = use_context::<EditorStore>();

    rsx! {
        div {
            style: "display:flex;flex-direction:column;align-items:center;\
                    gap:14px;margin-top:16px;min-width:320px;\
                    padding:24px 28px;background:#262626;\
                    border:1px solid #3c3c3c;border-radius:8px;",

            // Spinner + project name on one row.
            div {
                style: "display:flex;align-items:center;gap:10px;",
                Loader { r#type: "oval", size: "sm", color: "blue" }
                div {
                    style: "font-size:13px;color:#ccc;",
                    {move || {
                        store.project_loading.get()
                            .map(|s| format!("Opening {}", s.project_name))
                            .unwrap_or_default()
                    }}
                }
            }

            // Phase label + optional detail.
            div {
                style: "font-size:12px;color:#888;text-align:center;",
                {move || {
                    store.project_loading.get()
                        .map(|s| s.phase.label().to_string())
                        .unwrap_or_default()
                }}
            }
            div {
                style: "font-size:11px;color:#666;text-align:center;\
                        font-family:monospace;",
                {move || {
                    store.project_loading.get()
                        .and_then(|s| s.detail)
                        .unwrap_or_default()
                }}
            }
        }
    }
}

#[component]
fn RecentProjectRow(project: RecentProject) -> NodeHandle {
    let store = use_context::<EditorStore>();
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
                let name = name.clone();
                let cmd = cmd.clone();
                move || {
                    if exists {
                        store.project_loading.send(Some(ProjectLoadingStatus {
                            project_name: name.clone(),
                            phase: ProjectLoadPhase::ScaffoldGameplay,
                            detail: None,
                        }));
                        let _ = cmd.0.send(arvx_engine::EngineCommand::OpenProject {
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
