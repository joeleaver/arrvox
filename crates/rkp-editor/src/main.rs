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
            })
        },
    );

    // 4. Spawn a test primitive.
    engine.send(rkp_engine::EngineCommand::SpawnPrimitive {
        name: "test_box".into(),
    });

    // 5. Run rinch UI.
    let cmd_tx = engine.cmd_tx.clone();

    let props = WindowProps {
        title: "RKIPatch Editor".into(),
        width: 1920,
        height: 1080,
        borderless: false,
        resizable: true,
        ..Default::default()
    };

    let theme = Some(ThemeProviderProps {
        dark_mode: true,
        primary_color: Some("blue".into()),
        ..Default::default()
    });

    rinch::shell::run_with_window_props(
        move |__scope| {
            create_context(surface_handle.clone());
            create_context(store);
            create_context(CommandSender(cmd_tx.clone()));
            rsx! { LayoutRoot {} }
        },
        props,
        theme,
    );

    drop(engine);
    Ok(())
}
