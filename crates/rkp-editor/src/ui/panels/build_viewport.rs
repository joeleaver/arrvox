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
use rinch::render_surface::{RenderSurface, SurfaceEvent, SurfaceMouseButton};

use rkp_engine::viewport::ViewportId;
use rkp_render::RenderMode;

use crate::{BuildSurface, CommandSender};
use crate::ui::store::EditorStore;
use super::viewport_toolbar::EditModeToolbar;

const PANEL_VIEWPORT: ViewportId = ViewportId::BUILD;

/// Turntable camera state — orbit about a target at a given distance.
/// The target is NOT a field here: it's pulled live from the selected
/// entity's Transform position each frame so the preview auto-tracks
/// when the procedural is moved in the scene. `eye(target)` computes
/// the orbit eye relative to whatever world target the caller resolved.
#[derive(Clone, Copy)]
struct Turntable {
    yaw: f32,
    pitch: f32,
    distance: f32,
    fov: f32,
}

impl Default for Turntable {
    fn default() -> Self {
        Self {
            yaw: 0.7,
            pitch: -0.3,
            distance: 4.0,
            fov: 50.0,
        }
    }
}

impl Turntable {
    /// Orbit eye for the given target, derived from yaw/pitch/distance.
    fn eye(&self, target: glam::Vec3) -> glam::Vec3 {
        let dir = glam::Vec3::new(
            -self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            -self.yaw.cos() * self.pitch.cos(),
        );
        target - dir * self.distance
    }
}

#[component]
pub fn BuildViewport() -> NodeHandle {
    let BuildSurface(surface) = use_context::<BuildSurface>();
    let cmd = use_context::<CommandSender>();
    let store = use_context::<EditorStore>();

    let turntable = Signal::new(Turntable::default());
    let mode = Signal::new(RenderMode::Isolation);
    let last_mx = std::cell::Cell::new(0.0f32);
    let last_my = std::cell::Cell::new(0.0f32);
    let orbiting = std::cell::Cell::new(false);

    // Push the mode to the engine whenever it changes. Edge-only via
    // a non-reactive Cell — same pattern as the visibility effect to
    // avoid re-queueing into a flush already in progress.
    {
        let prev_mode = std::cell::Cell::new(None::<RenderMode>);
        let cmd_tx = cmd.0.clone();
        __scope.create_effect(move || {
            let m = mode.get();
            if prev_mode.get() == Some(m) {
                return;
            }
            prev_mode.set(Some(m));
            let _ = cmd_tx.send(rkp_engine::EngineCommand::SetViewportMode {
                id: PANEL_VIEWPORT,
                mode: m,
            });
        });
    }

    // Whenever `procedural` flips between Some and None, drive the
    // viewport's visibility. `__scope.create_effect` ties the effect's
    // lifetime to the component — it gets disposed on unmount rather
    // than leaking past tab switches.
    //
    // Critical: effects must NEVER write to signals that any co-running
    // effect on the same dep chain reads. store.procedural's state-tick
    // updates push all subscribers into the flush queue; if we wrote to
    // `turntable` here, the nested flush would pop the has_procedural
    // memo marker and re-queue this effect while it's still borrowed —
    // that's the RefCell-already-borrowed panic we've chased twice now.
    // Hence: no turntable mutation in this effect. If the user wants a
    // re-frame, they can trigger it from a button or a drag-to-refocus.
    let has_procedural = Memo::new(move || store.procedural.get().is_some());
    {
        // `store.procedural.send` is called unconditionally on every
        // state-tick from the engine thread, so this effect fires once
        // per frame even when visibility hasn't actually changed. Track
        // the previous value in a non-reactive Cell (a Signal here would
        // re-queue this effect, see the panic note above) and only send
        // on the actual edge.
        let prev_visible = std::cell::Cell::new(None::<bool>);
        let cmd_tx = cmd.0.clone();
        __scope.create_effect(move || {
            let visible = has_procedural.get();
            if prev_visible.get() == Some(visible) {
                return;
            }
            prev_visible.set(Some(visible));
            let _ = cmd_tx.send(rkp_engine::EngineCommand::SetViewportVisible {
                id: PANEL_VIEWPORT,
                visible,
            });
        });
    }

    // Focus the viewport on whichever procedural entity is currently
    // selected. The build viewport's default filter is `BUILD_PREVIEW`
    // which excludes the `DEFAULT` bit all normal entities carry —
    // focus_entity is the additive escape hatch that lets the selected
    // procedural through the visibility gate regardless of its layer.
    // Without this, the build viewport renders an empty sky.
    {
        let prev_focus = std::cell::Cell::new(None::<Option<uuid::Uuid>>);
        let cmd_tx = cmd.0.clone();
        __scope.create_effect(move || {
            let focus = store.selected_entity.get();
            if prev_focus.get() == Some(focus) {
                return;
            }
            prev_focus.set(Some(focus));
            let _ = cmd_tx.send(rkp_engine::EngineCommand::SetViewportFilter {
                id: PANEL_VIEWPORT,
                base_layers: rkp_engine::viewport::layer::BUILD_PREVIEW,
                focus_entity_id: focus,
            });
        });
    }

    // Push the current turntable pose to the engine whenever the orbit
    // parameters change OR the selected entity's world position changes.
    // Target tracks the inspector's `position` so gizmo-translating the
    // procedural in the main viewport re-centers the build preview in
    // lock-step — no manual "re-frame" button needed.
    //
    // Effect reads: `turntable`, `store.inspector`. Writes: SetCamera only.
    // Does NOT write into `turntable` (that would re-queue this effect
    // against the state-tick's signal flush — see the RefCell panic
    // note further up).
    {
        let cmd_tx = cmd.0.clone();
        __scope.create_effect(move || {
            let t = turntable.get();
            let target = store.inspector.get()
                .map(|i| glam::Vec3::from(i.position))
                .unwrap_or(glam::Vec3::ZERO);
            let eye = t.eye(target);
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
    // Track the last dispatched size so we only fire Resize on actual
    // changes (every event fires the handler; don't spam the channel).
    let last_size = std::cell::Cell::new((0u32, 0u32));
    // Remember the last mouse position so we can compute deltas for
    // MouseMove commands (the panel already tracks `last_mx`/`last_my`
    // for orbit math but those would leak across left/middle/right).
    surface.set_event_handler(move |event| {
        use SurfaceEvent::*;
        use rkf_runtime::input::InputMouseButton;

        // Map a rinch button to the runtime enum the engine expects.
        fn map_btn(b: SurfaceMouseButton) -> InputMouseButton {
            match b {
                SurfaceMouseButton::Left => InputMouseButton::Left,
                SurfaceMouseButton::Right => InputMouseButton::Right,
                SurfaceMouseButton::Middle => InputMouseButton::Middle,
                _ => InputMouseButton::Left,
            }
        }

        // Relay panel size to the engine so BUILD's VR renders at the
        // panel's native resolution. Each VR has its own pass chain
        // now (Phase 6 pass-internal split), so this doesn't clobber
        // MAIN's resources.
        {
            let (w, h) = surface_for_handler.layout_size();
            let w = w.max(64);
            let h = h.max(64);
            if last_size.get() != (w, h) {
                last_size.set((w, h));
                let _ = cmd_tx.send(rkp_engine::EngineCommand::Resize {
                    id: PANEL_VIEWPORT, width: w, height: h,
                });
            }
        }

        match event {
            MouseDown { button, x, y } => {
                last_mx.set(x);
                last_my.set(y);
                // Right-drag (and middle) orbits, matching the main
                // viewport's right-drag-to-look convention. Left-click
                // is forwarded to the engine for gizmo picking.
                if button == SurfaceMouseButton::Right
                    || button == SurfaceMouseButton::Middle
                {
                    orbiting.set(true);
                }
                let _ = cmd_tx.send(rkp_engine::EngineCommand::MouseButton {
                    id: PANEL_VIEWPORT,
                    button: map_btn(button),
                    pressed: true,
                });
            }
            MouseUp { button, .. } => {
                orbiting.set(false);
                let _ = cmd_tx.send(rkp_engine::EngineCommand::MouseButton {
                    id: PANEL_VIEWPORT,
                    button: map_btn(button),
                    pressed: false,
                });
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
                // Always forward movement — the engine needs live
                // cursor position for gizmo hover + drag, independent
                // of the local orbit state.
                let _ = cmd_tx.send(rkp_engine::EngineCommand::MouseMove {
                    id: PANEL_VIEWPORT, x, y, dx, dy,
                });
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

    // Layout mirrors the main Viewport panel: flex-column with a
    // flex:1+min-height:0+position:relative content area so the
    // RenderSurface can size to the container. The placeholder is an
    // absolute overlay instead of a conditional mount — re-mounting
    // `RenderSurface` each time `has_procedural` flips would reset the
    // panel's surface attachment on every selection change.
    rsx! {
        div {
            style: "display:flex;flex-direction:column;width:100%;height:100%;\
                    background:#1a1a1a;",
            // Header row with isolation/in-situ toggle. Sized to match
            // the main viewport's ViewportHeaderBar so panels align.
            div {
                style: "height:32px;display:flex;align-items:center;\
                        padding:0 6px;background:#252526;\
                        border-bottom:1px solid #3c3c3c;flex-shrink:0;gap:4px;",
                {mode_toggle(__scope, mode, RenderMode::Isolation, "Isolation",
                             "Studio backdrop with grid")}
                {mode_toggle(__scope, mode, RenderMode::InSitu, "In-Situ",
                             "Match scene environment")}
            }
            div {
                style: "flex:1;min-height:0;position:relative;",
                RenderSurface { surface: Some(surface.clone()) }
                // Floating gizmo-mode toolbar. Shares gizmo_mode state
                // with MAIN's EditModeToolbar — toggling here flips
                // both viewports' gizmos. Only meaningful while a
                // procedural is open.
                if has_procedural.get() {
                    EditModeToolbar {}
                }
                if !has_procedural.get() {
                    div {
                        style: "position:absolute;inset:0;display:flex;\
                                align-items:center;justify-content:center;\
                                background:#1a1a1a;color:#666;font-style:italic;\
                                font-size:12px;",
                        "Select a procedural object to build"
                    }
                }
            }
        }
    }
}

/// Segmented-button entry. `current` is the live mode signal; clicking
/// sets it to `value`. The active variant gets a brighter background.
fn mode_toggle(
    __scope: &mut rinch::core::dom::RenderScope,
    current: Signal<RenderMode>,
    value: RenderMode,
    label: &'static str,
    title: &'static str,
) -> rinch::core::dom::NodeHandle {
    rsx! {
        div {
            style: {move || {
                if current.get() == value {
                    "padding:2px 10px;background:#3c5a8a;border:1px solid #4a78b0;\
                     border-radius:4px;cursor:pointer;color:#dde7f5;font-size:11px;\
                     font-weight:600;"
                } else {
                    "padding:2px 10px;background:#2a2a2a;border:1px solid #3c3c3c;\
                     border-radius:4px;cursor:pointer;color:#a0a0a0;font-size:11px;"
                }
            }},
            title: title,
            onclick: move || {
                if current.get() != value {
                    current.set(value);
                }
            },
            {label}
        }
    }
}
