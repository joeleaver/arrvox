//! Build viewport — live preview of the selected procedural object.
//!
//! Renders into its own `RenderSurface` keyed to `ViewportId::BUILD`.
//! Driven by a turntable camera (Alt+drag to orbit, scroll to zoom)
//! that the editor computes and pushes to the engine via SetCamera.
//! Visibility follows `store.procedural.is_some()` — opening a
//! procedural flips the viewport on, deselecting flips it off.
//!
//! The tree + params UI from the original build panel overlays this
//! surface as a transparent strip down one side; the 3D view occupies
//! the rest of the panel.

use rinch::prelude::*;
// Effect isn't in the prelude — it's reserved for sync with external
// systems (command channel, here). DOM reactivity uses {|| expr}.
use rinch::core::Effect;
use rinch::render_surface::{RenderSurface, SurfaceEvent, SurfaceMouseButton};

use rkp_engine::viewport::ViewportId;

use crate::{BuildSurface, CommandSender};
use crate::ui::store::EditorStore;

const PANEL_VIEWPORT: ViewportId = ViewportId::BUILD;

/// Turntable camera state — orbit about a target at a given distance.
/// Converted to `SetCamera { position, yaw, pitch, fov }` every time the
/// user drags or scrolls; the engine stores that as the viewport's
/// `editor_camera` and renders from it on the next tick.
#[derive(Clone, Copy)]
struct Turntable {
    target: glam::Vec3,
    yaw: f32,
    pitch: f32,
    distance: f32,
    fov: f32,
}

impl Default for Turntable {
    fn default() -> Self {
        Self {
            target: glam::Vec3::ZERO,
            yaw: 0.7,
            pitch: -0.3,
            distance: 4.0,
            fov: 50.0,
        }
    }
}

impl Turntable {
    /// Current orbit position, derived from yaw/pitch + distance.
    fn eye(&self) -> glam::Vec3 {
        let dir = glam::Vec3::new(
            -self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            -self.yaw.cos() * self.pitch.cos(),
        );
        self.target - dir * self.distance
    }
}

#[component]
pub fn BuildViewport() -> NodeHandle {
    let BuildSurface(surface) = use_context::<BuildSurface>();
    let cmd = use_context::<CommandSender>();
    let store = use_context::<EditorStore>();

    let turntable = Signal::new(Turntable::default());
    let last_mx = std::cell::Cell::new(0.0f32);
    let last_my = std::cell::Cell::new(0.0f32);
    let orbiting = std::cell::Cell::new(false);

    // Whenever `procedural` flips between Some and None, drive the
    // viewport's visibility. Use a Memo so the effect only fires on the
    // actual transition, not every unrelated store update.
    let has_procedural = Memo::new(move || store.procedural.get().is_some());
    {
        let cmd_tx = cmd.0.clone();
        Effect::new(move || {
            let visible = has_procedural.get();
            let _ = cmd_tx.send(rkp_engine::EngineCommand::SetViewportVisible {
                id: PANEL_VIEWPORT,
                visible,
            });
            // Re-seed the turntable target on each open. Phase 7 can
            // center on the procedural's actual AABB; MVP uses origin.
            if visible {
                turntable.update(|t| *t = Turntable::default());
            }
        });
    }

    // Push the current turntable pose to the engine whenever it changes.
    // The engine stores it on the BUILD viewport's editor_camera and
    // render_frame reads from there next tick.
    {
        let cmd_tx = cmd.0.clone();
        Effect::new(move || {
            let t = turntable.get();
            let eye = t.eye();
            let _ = cmd_tx.send(rkp_engine::EngineCommand::SetCamera {
                id: PANEL_VIEWPORT,
                position: eye,
                yaw: t.yaw,
                pitch: t.pitch,
                fov: t.fov,
            });
        });
    }

    let cmd_tx = cmd.0.clone();
    let surface_for_handler = surface.clone();
    surface.set_event_handler(move |event| {
        use SurfaceEvent::*;

        // Relay size changes to the engine so the VR resizes its
        // gbuffer + bloom chain to match.
        {
            let (w, h) = surface_for_handler.layout_size();
            let w = w.max(64);
            let h = h.max(64);
            let _ = cmd_tx.send(rkp_engine::EngineCommand::Resize {
                id: PANEL_VIEWPORT, width: w, height: h,
            });
        }

        match event {
            MouseDown { button, x, y } => {
                last_mx.set(x);
                last_my.set(y);
                if button == SurfaceMouseButton::Left
                    || button == SurfaceMouseButton::Middle
                {
                    orbiting.set(true);
                }
            }
            MouseUp { .. } => {
                orbiting.set(false);
            }
            MouseMove { x, y } => {
                let dx = x - last_mx.get();
                let dy = y - last_my.get();
                last_mx.set(x);
                last_my.set(y);
                if orbiting.get() {
                    turntable.update(|t| {
                        t.yaw -= dx * 0.005;
                        t.pitch = (t.pitch - dy * 0.005)
                            .clamp(-1.5, 1.5); // stop at +/- ~85°
                    });
                }
            }
            MouseWheel { delta_y, .. } => {
                // Scroll zooms in/out of the target.
                turntable.update(|t| {
                    let scale = if delta_y > 0.0 { 0.9 } else { 1.1 };
                    t.distance = (t.distance * scale).clamp(0.1, 200.0);
                });
            }
            _ => {}
        }
    });

    rsx! {
        div {
            style: "position:relative;width:100%;height:100%;background:#1a1a1a;",
            // No procedural selected → placeholder text. The viewport
            // itself is hidden engine-side via SetViewportVisible, so
            // the surface doesn't get frames.
            if !has_procedural.get() {
                div {
                    style: "display:flex;align-items:center;justify-content:center;\
                            height:100%;color:#666;font-style:italic;font-size:12px;",
                    "Select a procedural object to build"
                }
            }
            if has_procedural.get() {
                RenderSurface { surface: Some(surface.clone()) }
            }
        }
    }
}
