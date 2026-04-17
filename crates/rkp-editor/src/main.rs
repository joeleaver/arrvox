//! RKIPatch Editor — thin client over the RkpEngine.
//!
//! Creates an RkpEngine on its own thread and a rinch UI as a thin client.
//! All UI state flows through `EditorStore`. The engine pushes state updates
//! via `send()`. The editor reads them reactively.

mod ui;

use rinch::prelude::*;

use ui::layout::persist::PersistedEditorState;
use ui::store::EditorStore;
use ui::LayoutRoot;

/// Snapshot the store's layout + widths and send them to the engine.
/// Called just before any Save command so the cached JSON on the
/// engine-side matches what the user sees when the project file is
/// written. Short-circuits if serialization fails (can't happen with
/// the current layout types, but we don't want a panic in the Save
/// hotkey path even in theory).
fn push_editor_layout(
    tx: &crossbeam::channel::Sender<rkp_engine::EngineCommand>,
    store: EditorStore,
) {
    let state = PersistedEditorState::capture(store);
    match serde_json::to_string(&state) {
        Ok(json) => {
            let _ = tx.send(rkp_engine::EngineCommand::SetEditorLayout { json });
        }
        Err(e) => eprintln!("[rkp-editor] failed to serialize editor layout: {e}"),
    }
}

/// Wrapper for the engine command sender, stored in rinch context.
#[derive(Clone)]
pub struct CommandSender(pub crossbeam::channel::Sender<rkp_engine::EngineCommand>);

/// Context newtype for the build viewport's render surface handle. The
/// bare `RenderSurfaceHandle` context slot is already taken by the main
/// viewport; distinguishing via newtype lets components ask for exactly
/// the surface they care about.
#[derive(Clone)]
pub struct BuildSurface(pub rinch::render_surface::RenderSurfaceHandle);

fn build_menus(
    cmd_tx: crossbeam::channel::Sender<rkp_engine::EngineCommand>,
    store: EditorStore,
) -> Vec<(&'static str, rinch::menu::Menu)> {
    use rinch::menu::{Menu, MenuItem};

    let tx = cmd_tx;

    // File menu
    let file_menu = Menu::new()
        .item(MenuItem::new("New Project").on_click({
            let tx = tx.clone();
            move || {
                if let Some(path) = rfd::FileDialog::new()
                    .set_title("New Project")
                    .add_filter("RKIPatch Project", &["rkproject"])
                    .save_file()
                {
                    let _ = tx.send(rkp_engine::EngineCommand::NewProject {
                        path: path.to_string_lossy().into_owned(),
                    });
                }
            }
        }))
        .item(MenuItem::new("Open Project...").on_click({
            let tx = tx.clone();
            move || {
                if let Some(path) = rfd::FileDialog::new()
                    .set_title("Open Project")
                    .add_filter("RKIPatch Project", &["rkproject"])
                    .pick_file()
                {
                    let _ = tx.send(rkp_engine::EngineCommand::OpenProject {
                        path: path.to_string_lossy().into_owned(),
                    });
                }
            }
        }))
        .separator()
        .item(MenuItem::new("Import Asset...").on_click({
            let tx = tx.clone();
            move || {
                if let Some(path) = rfd::FileDialog::new()
                    .set_title("Import Mesh Asset")
                    .add_filter("3D Models", &["glb", "gltf", "obj", "fbx"])
                    .pick_file()
                {
                    let _ = tx.send(rkp_engine::EngineCommand::ImportAsset {
                        source_path: path.to_string_lossy().into_owned(),
                    });
                }
            }
        }))
        .separator()
        .item(MenuItem::new("Save").shortcut("Ctrl+S").on_click({
            let tx = tx.clone();
            move || {
                // Push the layout FIRST so the engine's cached blob is
                // fresh when the SaveScene handler folds it into the
                // project file. Commands are FIFO on the channel, so
                // ordering here is the ordering the engine sees.
                push_editor_layout(&tx, store);
                let _ = tx.send(rkp_engine::EngineCommand::SaveScene { path: None });
            }
        }))
        .item(MenuItem::new("Save As...").shortcut("Ctrl+Shift+S").on_click({
            let tx = tx.clone();
            move || {
                if let Some(path) = rfd::FileDialog::new()
                    .set_title("Save Scene As")
                    .add_filter("RKIPatch Scene", &["rkscene"])
                    .save_file()
                {
                    push_editor_layout(&tx, store);
                    let _ = tx.send(rkp_engine::EngineCommand::SaveScene {
                        path: Some(path.to_string_lossy().into_owned()),
                    });
                }
            }
        }));

    // Edit menu — spawn procedural primitives. Each item creates an entity
    // with a Root → <leaf> tree; shape parameters are editable afterwards
    // in the build panel.
    let mut spawn_menu = Menu::new();
    for leaf in ["Sphere", "Box", "Capsule", "Cylinder", "Torus", "Ramp"] {
        spawn_menu = spawn_menu.item(MenuItem::new(leaf).on_click({
            let tx = tx.clone();
            move || {
                let _ = tx.send(rkp_engine::EngineCommand::SpawnProceduralObject {
                    name: leaf.to_string(),
                    leaf_kind: Some(leaf.to_string()),
                });
            }
        }));
    }

    spawn_menu = spawn_menu
        .separator()
        .item(MenuItem::new("Point Light").on_click({
            let tx = tx.clone();
            move || { let _ = tx.send(rkp_engine::EngineCommand::SpawnPointLight); }
        }))
        .item(MenuItem::new("Spot Light").on_click({
            let tx = tx.clone();
            move || { let _ = tx.send(rkp_engine::EngineCommand::SpawnSpotLight); }
        }))
        .separator()
        .item(MenuItem::new("Camera").on_click({
            let tx = tx.clone();
            move || { let _ = tx.send(rkp_engine::EngineCommand::SpawnCamera); }
        }));

    let edit_menu = Menu::new()
        .submenu("Spawn", spawn_menu)
        .separator()
        .item(MenuItem::new("Duplicate").shortcut("Ctrl+D").on_click({
            let tx = tx.clone();
            move || {
                let _ = tx.send(rkp_engine::EngineCommand::DuplicateSelected);
            }
        }))
        .item(MenuItem::new("Delete").shortcut("Delete").on_click({
            let tx = tx.clone();
            move || {
                let _ = tx.send(rkp_engine::EngineCommand::DeleteSelected);
            }
        }));

    let view_menu = Menu::new()
        .item(MenuItem::new("Show Colliders").on_click({
            let tx = tx.clone();
            let toggle = std::cell::Cell::new(false);
            move || {
                let new_val = !toggle.get();
                toggle.set(new_val);
                let _ = tx.send(rkp_engine::EngineCommand::SetViewOption {
                    option: "show_colliders".into(),
                    enabled: new_val,
                });
            }
        }));

    vec![
        ("File", file_menu),
        ("Edit", edit_menu),
        ("View", view_menu),
    ]
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    // 1. Create render surfaces — one per viewport. MAIN renders the
    //    scene; BUILD renders the selected procedural object in its own
    //    panel.
    // Both surfaces go through rinch's inline-paint path (via the
    // `RenderSurface` component's `data-render-surface` attribute).
    // `create_render_surface_with_name` is for video/compositor
    // hole-punch and was the wrong pick — it would route BUILD's pixels
    // to a different path that needs a `data-viewport` DOM attribute we
    // don't emit, so BUILD would never actually paint.
    let main_surface_handle = create_render_surface();
    let main_surface_writer = main_surface_handle.writer();
    let build_surface_handle = create_render_surface();
    let build_surface_writer = build_surface_handle.writer();

    // 2. Create the central editor store. Surface handles travel via
    //    rinch context, not EditorStore (EditorStore is Copy; handles
    //    aren't).
    let store = EditorStore::new();

    // 3. Start the engine.
    let engine = rkp_engine::RkpEngine::spawn(
        rkp_engine::engine::EngineConfig {
            width: 1920,
            height: 1080,
        },
        // Frame callback: route each viewport's pixels to its writer.
        // Unknown viewport ids are silently dropped (defensive — the
        // engine only emits MAIN/BUILD today).
        Box::new(move |viewport_id, pixels, w, h| {
            use rkp_engine::viewport::ViewportId;
            if viewport_id == ViewportId::MAIN {
                main_surface_writer.submit_frame(pixels, w, h);
            } else if viewport_id == ViewportId::BUILD {
                build_surface_writer.submit_frame(pixels, w, h);
            }
        }),
        // State callback: push engine state to EditorStore signals (cross-thread).
        {
            let store = store;
            Box::new(move |update: &rkp_engine::StateUpdate| {
                store.fps.send(update.fps);
                store.gpu_object_count.send(update.gpu_object_count);
                store.selected_entity.send(update.selected_entity);
                if let Some(objects) = &update.objects {
                    store.objects.send(objects.clone());
                }
                if let Some(loaded) = update.project_loaded {
                    store.project_loaded.send(loaded);
                }
                if let Some(name) = &update.project_name {
                    store.project_name.send(name.clone());
                }
                if let Some(models) = &update.available_models {
                    store.available_models.send(models.clone());
                }
                if let Some(importing) = &update.importing_models {
                    store.importing_models.send(importing.clone());
                }
                if let Some(progress) = &update.import_progress {
                    store.import_progress.send(progress.clone());
                }
                // No explicit clear on idle — `get()` isn't allowed off
                // the main thread, and the UI gates progress-bar display
                // on `importing_models` anyway, so stale `import_progress`
                // entries sit harmlessly until the next import replaces
                // them. (If we ever need a clean-up, do it on the engine
                // side with a dirty flag and send `Some(Vec::new())` on
                // the empty→empty transition.)
                // Hydrate layout on project open. Outer Some = "this
                // tick carries a layout update"; inner None = "project
                // had none stored, reset to defaults". The engine only
                // sets this once per load; layouts edited in-session
                // live entirely on the editor side until the next save.
                if let Some(layout_payload) = &update.editor_layout {
                    let state = layout_payload
                        .as_deref()
                        .map(PersistedEditorState::from_json_or_default)
                        .unwrap_or_default();
                    // apply() uses Signal::send internally, which hops
                    // to the UI thread on its own.
                    state.apply(store);
                }
                store.inspector.send(update.inspector.clone());
                store.procedural.send(update.procedural.clone());
                if let Some(ref ac) = update.available_components {
                    store.available_components.send(ac.clone());
                }
                if let Some(ref rp) = update.recent_projects {
                    store.recent_projects.send(rp.clone());
                }
                if let Some(ref mats) = update.materials {
                    store.materials.send(mats.clone());
                }
                // Mirror the Option as-is — a None from the engine means
                // "nothing selected", and ignoring it (old behavior) left
                // the Asset Properties panel stuck on the previous pick
                // after the engine swapped selection from material→model
                // or vice versa.
                store.selected_material.send(update.selected_material);
                store.selected_model.send(update.selected_model.clone());
                store.play_mode.send(update.play_mode);
                if let Some(ref env) = update.environment {
                    store.environment.send(env.clone());
                }
                if !update.console_entries.is_empty() {
                    let new_entries = update.console_entries.clone();
                    store.console_entries.update_send(move |entries| {
                        entries.extend(new_entries);
                        // Cap at 1000 entries in the UI.
                        if entries.len() > 1000 {
                            let excess = entries.len() - 1000;
                            entries.drain(..excess);
                        }
                    });
                }
            })
        },
    );

    // 5. Build menus.
    let menus = build_menus(engine.cmd_tx.clone(), store);

    // 6. Run rinch UI.
    let cmd_tx = engine.cmd_tx.clone();

    let props = WindowProps {
        title: "RKIPatch Editor".into(),
        width: 1920,
        height: 1080,
        borderless: true,
        resizable: true,
        transparent: true,
        menu_in_titlebar: true,
        ..Default::default()
    };

    let theme = Some(ThemeProviderProps {
        dark_mode: true,
        primary_color: Some("blue".into()),
        ..Default::default()
    });

    rinch::shell::run_with_window_props_and_menu(
        move |__scope| {
            create_context(main_surface_handle.clone());
            create_context(BuildSurface(build_surface_handle.clone()));
            create_context(store);
            create_context(CommandSender(cmd_tx.clone()));
            rsx! { LayoutRoot {} }
        },
        props,
        theme,
        Some(menus),
    );

    drop(engine);
    Ok(())
}
