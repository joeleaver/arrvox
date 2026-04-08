//! RKIPatch Editor — thin client over the RkpEngine.
//!
//! Creates an RkpEngine on its own thread and a rinch UI as a thin client.
//! All UI state flows through `EditorStore`. The engine pushes state updates
//! via `send()`. The editor reads them reactively.

mod ui;

use rinch::prelude::*;

use ui::store::EditorStore;
use ui::LayoutRoot;

/// Wrapper for the engine command sender, stored in rinch context.
#[derive(Clone)]
pub struct CommandSender(pub crossbeam::channel::Sender<rkp_engine::EngineCommand>);

fn build_menus(
    cmd_tx: crossbeam::channel::Sender<rkp_engine::EngineCommand>,
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
                    let _ = tx.send(rkp_engine::EngineCommand::SaveScene {
                        path: Some(path.to_string_lossy().into_owned()),
                    });
                }
            }
        }));

    // Edit menu — spawn primitives
    let mut spawn_menu = Menu::new();
    for (label, prim_name) in [("Box", "box"), ("Sphere", "sphere"), ("Capsule", "capsule")] {
        spawn_menu = spawn_menu.item(MenuItem::new(label).on_click({
            let tx = tx.clone();
            let name = prim_name.to_string();
            move || {
                let _ = tx.send(rkp_engine::EngineCommand::SpawnPrimitive {
                    name: name.clone(),
                });
            }
        }));
    }

    let edit_menu = Menu::new()
        .submenu("Spawn", spawn_menu);

    vec![
        ("File", file_menu),
        ("Edit", edit_menu),
    ]
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    // 1. Create render surface.
    let surface_handle = create_render_surface();
    let surface_writer = surface_handle.writer();

    // 2. Create the central editor store.
    let store = EditorStore::new();

    // 3. Start the engine.
    let engine = rkp_engine::RkpEngine::spawn(
        rkp_engine::engine::EngineConfig {
            width: 1920,
            height: 1080,
        },
        // Frame callback: deliver pixels to rinch surface.
        Box::new(move |pixels, w, h| {
            surface_writer.submit_frame(pixels, w, h);
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
                store.inspector.send(update.inspector.clone());
                if let Some(ref ac) = update.available_components {
                    store.available_components.send(ac.clone());
                }
            })
        },
    );

    // 5. Build menus.
    let menus = build_menus(engine.cmd_tx.clone());

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
            create_context(surface_handle.clone());
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
