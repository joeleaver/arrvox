//! RKIPatch Editor — thin client over the RkpEngine.
//!
//! Creates an RkpEngine on its own thread and a rinch UI as a thin client.
//! The engine pushes state updates via signals. The editor reads them reactively.

mod ui;

use rinch::prelude::*;

use rkp_engine::SceneObjectInfo;
use ui::EditorUi;

/// Signals the engine writes to (via `send()`), the UI reads (via `get()`).
///
/// All fields are `Signal` (Copy). Created before the engine and UI start,
/// shared by both via rinch context.
#[derive(Clone, Copy)]
pub struct EngineSignals {
    pub fps: Signal<f32>,
    pub gpu_object_count: Signal<u32>,
    pub objects: Signal<Vec<SceneObjectInfo>>,
    pub selected_entity: Signal<Option<uuid::Uuid>>,
}

/// Wrapper for the engine command sender, stored in rinch context.
#[derive(Clone)]
pub struct CommandSender(pub crossbeam::channel::Sender<rkp_engine::EngineCommand>);

fn main() -> anyhow::Result<()> {
    env_logger::init();

    // 1. Create render surface.
    let surface_handle = create_render_surface();
    let surface_writer = surface_handle.writer();

    // 2. Create signals for engine→UI communication.
    let signals = EngineSignals {
        fps: Signal::new(0.0),
        gpu_object_count: Signal::new(0),
        objects: Signal::new(Vec::new()),
        selected_entity: Signal::new(None),
    };

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
        // State callback: push engine state to UI signals (cross-thread via send()).
        {
            let signals = signals;
            Box::new(move |update: &rkp_engine::StateUpdate| {
                signals.fps.send(update.fps);
                signals.gpu_object_count.send(update.gpu_object_count);
                signals.selected_entity.send(update.selected_entity);
                if let Some(objects) = &update.objects {
                    signals.objects.send(objects.clone());
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
            create_context(signals);
            create_context(CommandSender(cmd_tx.clone()));
            rsx! { EditorUi {} }
        },
        props,
        theme,
    );

    drop(engine);
    Ok(())
}
