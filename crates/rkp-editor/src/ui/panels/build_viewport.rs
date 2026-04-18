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
use rkp_render::BuildPreviewMode;

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
    let last_mx = std::cell::Cell::new(0.0f32);
    let last_my = std::cell::Cell::new(0.0f32);
    let orbiting = std::cell::Cell::new(false);

    // BUILD always uses Isolation rendering (studio backdrop + grid).
    // The mode is set once at viewport creation in `viewport.rs`; no
    // toggle exists here because the old In-Situ option added a lot
    // of pass-chain complexity for almost no editing benefit — the
    // MAIN viewport already shows the in-scene look.

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

    // Drive the viewport's visibility filter from the selected entity.
    // Always `BUILD_PREVIEW` base — excludes normal scene objects;
    // `focus_entity` is the additive escape hatch that lets the
    // selected procedural through regardless of its layer bit.
    {
        let prev = std::cell::Cell::new(None::<Option<uuid::Uuid>>);
        let cmd_tx = cmd.0.clone();
        __scope.create_effect(move || {
            let focus = store.selected_entity.get();
            if prev.get() == Some(focus) {
                return;
            }
            prev.set(Some(focus));
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
                // Top-right: preview-mode toggle + Bake action + resolution.
                // Floats as a single compact overlay over the 3D view so
                // the tools live next to the thing they modify. Gated on
                // `has_procedural` — same as the tree panel.
                if has_procedural.get() {
                    div {
                        style: "position:absolute;top:8px;right:8px;\
                                background:rgba(30,30,30,0.88);\
                                border:1px solid #3c3c3c;border-radius:4px;\
                                color:#ccc;font-size:12px;padding:6px;\
                                backdrop-filter:blur(4px);z-index:15;\
                                display:flex;flex-direction:column;gap:6px;\
                                min-width:240px;",
                        {render_preview_toggle(__scope, store.clone(), tree_cmd_tx)}
                        {render_resolution(__scope, tree_snapshot, tree_cmd_tx)}
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

/// Bake button + voxel-count readout + dirty indicator. Interactive
/// edits mark the tree dirty but don't rebake — this button pays the
/// voxelization cost on demand. Highlights blue when there are unbaked
/// changes. The readout shows the last bake's voxel count, which is
/// always worth eyeballing: resolution changes can easily 100× the
/// output without the user noticing until stepping through perf.
fn render_bake_action(
    __scope: &mut rinch::core::dom::RenderScope,
    snapshot: Memo<ProceduralSnapshot>,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> rinch::core::dom::NodeHandle {
    let dirty = Memo::new(move || snapshot.get().dirty);
    let voxel_count = Memo::new(move || snapshot.get().voxel_count);
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
            div {
                style: "display:flex;flex-direction:column;align-items:flex-end;\
                        line-height:1.2;",
                span {
                    style: "font-size:11px;color:#999;font-family:monospace;",
                    {move || format_voxel_count_readout(voxel_count.get())}
                }
                span {
                    style: {move || if dirty.get() {
                        "font-size:10px;color:#f0a04b;"
                    } else {
                        "font-size:10px;color:#555;"
                    }},
                    {move || if dirty.get() { "unbaked" } else { "up to date" }}
                }
            }
        }
    }
}

/// Resolution picker — tier buffer size for the voxelizer. Same dropdown
/// that used to live in the right panel; moved into the viewport overlay
/// so the Bake / Voxel-count / Resolution controls all sit together,
/// since they form one logical workflow (pick resolution → bake → glance
/// at count → adjust).
fn render_resolution(
    __scope: &mut rinch::core::dom::RenderScope,
    snapshot: Memo<ProceduralSnapshot>,
    cmd_tx: Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>,
) -> rinch::core::dom::NodeHandle {
    // Track the current voxel_size so the select reflects engine changes
    // (e.g., safe-voxel-size auto-bumps on large AABBs). `prop_select`
    // reads its initial value from the Signal at mount; a Memo keeps us
    // in sync on future snapshots without remounting the whole select.
    let current = Signal::new(format!("{}", snapshot.get().voxel_size));
    let on_change: std::rc::Rc<dyn Fn(String)> = std::rc::Rc::new(move |v: String| {
        let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SetProceduralVoxelSize {
            tier: v,
        });
    });
    super::prop_controls::prop_select(
        __scope,
        "Res",
        Memo::new(move || current.get()),
        // Same tuples the ProceduralGeometry `voxel_size` field picker
        // shows in the properties panel — one `PROCEDURAL_VOXEL_TIERS`
        // const drives both surfaces.
        rkp_engine::components::PROCEDURAL_VOXEL_TIERS,
        on_change,
    )
}

/// Format a voxel count for the single-line readout. "—" when the tree
/// hasn't been baked yet (count=0), compact SI-ish suffixes otherwise.
fn format_voxel_count_readout(count: u32) -> String {
    if count == 0 {
        "— voxels".to_string()
    } else if count >= 1_000_000 {
        format!("{:.1}M voxels", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}K voxels", count as f64 / 1_000.0)
    } else {
        format!("{count} voxels")
    }
}
