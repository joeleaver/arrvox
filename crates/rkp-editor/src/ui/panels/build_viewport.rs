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

use rkp_engine::procedural_snapshot::ProceduralSnapshot;
use rkp_engine::viewport::ViewportId;
use rkp_render::{BuildPreviewMode, RenderMode};

use crate::{BuildSurface, CommandSender};
use crate::ui::store::EditorStore;
use super::procedural_tree;
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

    // Signals for the floating procedural-tree overlay rendered on
    // top of the viewport surface. Mirrors the Memos the build panel
    // used when the tree lived there — the widget itself is
    // self-contained in `procedural_tree::render_tree`.
    let tree_cmd_tx = Signal::new(cmd.0.clone());
    let tree_snapshot = Memo::new(move || store.procedural.get().unwrap_or_default());
    let tree_selected_node = Memo::new(move || tree_snapshot.get().selected_node);
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

    // Drive the viewport's visibility filter from (selected_entity, mode).
    //
    // - Isolation: `BUILD_PREVIEW` base — excludes normal scene objects.
    //   focus_entity is the additive escape hatch that lets the selected
    //   procedural through regardless of its (DEFAULT) layer bit.
    // - In-Situ: `DEFAULT` base — the build preview sees whatever the
    //   main scene contains, so the procedural is rendered in context.
    //   focus_entity stays set so the selection remains visible even if
    //   somebody tagged it out of DEFAULT.
    {
        let prev = std::cell::Cell::new(None::<(Option<uuid::Uuid>, RenderMode)>);
        let cmd_tx = cmd.0.clone();
        __scope.create_effect(move || {
            let focus = store.selected_entity.get();
            let m = mode.get();
            if prev.get() == Some((focus, m)) {
                return;
            }
            prev.set(Some((focus, m)));
            let base_layers = match m {
                RenderMode::Isolation => rkp_engine::viewport::layer::BUILD_PREVIEW,
                RenderMode::InSitu => rkp_engine::viewport::layer::DEFAULT,
            };
            let _ = cmd_tx.send(rkp_engine::EngineCommand::SetViewportFilter {
                id: PANEL_VIEWPORT,
                base_layers,
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
                // Left click → pick. In raymarch preview mode the
                // engine decodes the hit pixel as a procedural
                // NodeId; in voxel mode the pick is a no-op on the
                // build viewport (there's only the one selected
                // procedural entity to "pick" and it's already
                // selected). Sent unconditionally to keep the event
                // path simple; the engine filters on preview_mode.
                if button == SurfaceMouseButton::Left {
                    let _ = cmd_tx.send(rkp_engine::EngineCommand::Pick {
                        id: PANEL_VIEWPORT,
                        x: x as u32,
                        y: y as u32,
                    });
                }
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
                // Floating procedural-tree overlay, anchored right of
                // the vertical gizmo-mode toolbar so they don't overlap
                // (toolbar is at top:8;left:8;z-index:20 — tree starts
                // at left:52 to clear a ~40px icon column with a bit
                // of breathing room).
                //
                // No `overflow` here: the add-child "+" popover
                // dropdown is an absolute-positioned descendant that
                // extends outside the tree's bounds; any overflow
                // setting on this container clips the dropdown and
                // it becomes invisible. If trees grow deep enough to
                // need scrolling we'll add an inner scroll container
                // that lives under the popover's own stacking, but
                // for now unbounded-height is fine.
                if has_procedural.get() {
                    div {
                        style: "position:absolute;top:8px;left:52px;width:240px;\
                                background:rgba(30,30,30,0.88);\
                                border:1px solid #3c3c3c;border-radius:4px;\
                                color:#ccc;font-size:12px;padding:4px;\
                                backdrop-filter:blur(4px);z-index:15;",
                        {procedural_tree::render_tree(
                            __scope, tree_snapshot, tree_selected_node, tree_cmd_tx,
                        )}
                    }
                }
                // Top-right: preview-mode toggle + Bake action. Floats as
                // a single compact overlay over the 3D view so the tools
                // live next to the thing they modify. Gated on
                // `has_procedural` — same as the tree panel.
                if has_procedural.get() {
                    div {
                        style: "position:absolute;top:8px;right:8px;\
                                background:rgba(30,30,30,0.88);\
                                border:1px solid #3c3c3c;border-radius:4px;\
                                color:#ccc;font-size:12px;padding:6px;\
                                backdrop-filter:blur(4px);z-index:15;\
                                display:flex;flex-direction:column;gap:6px;\
                                min-width:220px;",
                        {render_preview_toggle(__scope, store.clone(), tree_cmd_tx)}
                        {render_bake_action(__scope, tree_snapshot, tree_cmd_tx)}
                    }
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

/// Two-button segmented control for the build viewport's primary-
/// visibility source. `Voxel` shows the baked octree result; `Procedural`
/// evaluates the analytical CSG tree per pixel so edits are live. The
/// pair is the expected editing loop: edit with Procedural, Bake, confirm
/// with Voxel. Rendered as the left half of the top-right overlay.
fn render_preview_toggle(
    __scope: &mut rinch::core::dom::RenderScope,
    store: EditorStore,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> rinch::core::dom::NodeHandle {
    let mode = store.build_preview_mode;

    let set_voxel = move || {
        mode.set(BuildPreviewMode::Voxel);
        let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetBuildPreviewMode {
            mode: BuildPreviewMode::Voxel,
        });
    };
    let set_raymarch = move || {
        mode.set(BuildPreviewMode::Raymarch);
        let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetBuildPreviewMode {
            mode: BuildPreviewMode::Raymarch,
        });
    };

    let btn_style = |active: bool| -> &'static str {
        if active {
            "flex:1;padding:3px 10px;background:#3c5a8a;color:#dde7f5;\
             border:1px solid #4a78b0;border-radius:3px;cursor:pointer;\
             font-size:11px;font-weight:600;"
        } else {
            "flex:1;padding:3px 10px;background:#2a2a2a;color:#a0a0a0;\
             border:1px solid #3c3c3c;border-radius:3px;cursor:pointer;\
             font-size:11px;"
        }
    };

    rsx! {
        div {
            style: "display:flex;align-items:center;gap:6px;",
            span { style: "color:#888;font-size:11px;", "Preview:" }
            div {
                style: "display:flex;flex:1;gap:4px;",
                button {
                    style: {move || btn_style(matches!(mode.get(), BuildPreviewMode::Raymarch))},
                    onclick: set_raymarch,
                    "Procedural"
                }
                button {
                    style: {move || btn_style(matches!(mode.get(), BuildPreviewMode::Voxel))},
                    onclick: set_voxel,
                    "Voxel"
                }
            }
        }
    }
}

/// Bake button + dirty indicator. Interactive edits mark the tree dirty
/// but don't rebake — this button pays the voxelization cost on demand.
/// Highlights blue when there are unbaked changes.
fn render_bake_action(
    __scope: &mut rinch::core::dom::RenderScope,
    snapshot: Memo<ProceduralSnapshot>,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> rinch::core::dom::NodeHandle {
    let dirty = Memo::new(move || snapshot.get().dirty);
    let entity_id = Memo::new(move || snapshot.get().entity_id);

    let on_click = move || {
        let _ = cmd_tx.get().send(rkp_engine::EngineCommand::BakeProceduralEntity {
            entity_id: entity_id.get(),
        });
    };

    rsx! {
        div {
            style: "display:flex;align-items:center;gap:8px;",
            button {
                style: {move || if dirty.get() {
                    "flex:1;padding:4px 10px;background:#2a4e7a;color:#fff;\
                     border:1px solid #3a6ea6;border-radius:3px;cursor:pointer;\
                     font-weight:600;font-size:11px;"
                } else {
                    "flex:1;padding:4px 10px;background:#2a2a2a;color:#888;\
                     border:1px solid #444;border-radius:3px;cursor:pointer;\
                     font-size:11px;"
                }},
                onclick: on_click,
                "Bake"
            }
            span {
                style: {move || if dirty.get() {
                    "color:#f0a04b;font-size:11px;"
                } else {
                    "color:#666;font-size:11px;"
                }},
                {move || if dirty.get() { "unbaked changes" } else { "up to date" }}
            }
        }
    }
}
