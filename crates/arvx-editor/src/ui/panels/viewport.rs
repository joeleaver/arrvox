//! Viewport component — renders the engine's output and forwards input events.

use rinch::prelude::*;
use rinch::render_surface::{RenderSurface, SurfaceEvent, SurfaceMouseButton};

use arvx_engine::viewport::ViewportId;

use crate::CommandSender;
use crate::ui::store::EditorStore;
use super::viewport_toolbar::{ViewportHeaderBar, EditModeToolbar, BrushToolbar, TerrainToolbar};

/// The viewport id this panel renders. Phase 3: only the MAIN viewport has
/// a UI panel; the build viewport gets its own component in Phase 6.
const PANEL_VIEWPORT: ViewportId = ViewportId::MAIN;

/// Map rinch SurfaceMouseButton to arvx_runtime InputMouseButton.
fn map_button(btn: SurfaceMouseButton) -> arvx_runtime::input::InputMouseButton {
    match btn {
        SurfaceMouseButton::Left => arvx_runtime::input::InputMouseButton::Left,
        SurfaceMouseButton::Right => arvx_runtime::input::InputMouseButton::Right,
        SurfaceMouseButton::Middle => arvx_runtime::input::InputMouseButton::Middle,
    }
}

/// Map a rinch key code string to InputKeyCode.
fn map_key(code: &str) -> Option<arvx_runtime::input::InputKeyCode> {
    use arvx_runtime::input::InputKeyCode::*;
    Some(match code {
        "KeyA" => A, "KeyB" => B, "KeyC" => C, "KeyD" => D,
        "KeyE" => E, "KeyF" => F, "KeyG" => G, "KeyH" => H,
        "KeyI" => I, "KeyJ" => J, "KeyK" => K, "KeyL" => L,
        "KeyM" => M, "KeyN" => N, "KeyO" => O, "KeyP" => P,
        "KeyQ" => Q, "KeyR" => R, "KeyS" => S, "KeyT" => T,
        "KeyU" => U, "KeyV" => V, "KeyW" => W, "KeyX" => X,
        "KeyY" => Y, "KeyZ" => Z,
        "Digit0" => Num0, "Digit1" => Num1, "Digit2" => Num2,
        "Digit3" => Num3, "Digit4" => Num4, "Digit5" => Num5,
        "Digit6" => Num6, "Digit7" => Num7, "Digit8" => Num8,
        "Digit9" => Num9,
        "Space" => Space,
        "ShiftLeft" => ShiftLeft, "ShiftRight" => ShiftRight,
        "ControlLeft" => ControlLeft, "ControlRight" => ControlRight,
        "AltLeft" => AltLeft, "AltRight" => AltRight,
        "Tab" => Tab, "Escape" => Escape, "Enter" => Enter,
        "Backspace" => Backspace, "Delete" => Delete,
        "ArrowUp" => ArrowUp, "ArrowDown" => ArrowDown,
        "ArrowLeft" => ArrowLeft, "ArrowRight" => ArrowRight,
        "F1" => F1, "F2" => F2, "F3" => F3, "F4" => F4,
        "F5" => F5, "F6" => F6, "F7" => F7, "F8" => F8,
        "F9" => F9, "F10" => F10, "F11" => F11, "F12" => F12,
        _ => return None,
    })
}

#[component]
pub fn Viewport() -> NodeHandle {
    let surface = use_context::<RenderSurfaceHandle>();
    let cmd = use_context::<CommandSender>();
    let store = use_context::<EditorStore>();

    // Track last mouse position for computing deltas.
    let last_mx = std::cell::Cell::new(0.0f32);
    let last_my = std::cell::Cell::new(0.0f32);
    // LMB-held state — paint mode uses this to decide whether a
    // MouseMove event should fire another `PaintAtPixel` stamp.
    let lmb_held = std::cell::Cell::new(false);
    // Stroke-spacing state for the sculpt brush. We record the screen
    // pixel of the last sculpt stamp in the active stroke and only
    // emit a new stamp when the cursor has moved by at least a
    // brush-radius-scaled threshold from there. Matches Blender's
    // "Space" stroke method (default 10% of brush radius). Reset on
    // every MouseDown so each new click starts a fresh stroke at the
    // cursor.
    let last_sculpt_stamp_x = std::cell::Cell::new(f32::NEG_INFINITY);
    let last_sculpt_stamp_y = std::cell::Cell::new(f32::NEG_INFINITY);
    // Stroke spacing as a fraction of brush radius. Blender defaults
    // to 0.10 for sculpt brushes; we use the same so the feel
    // matches.
    const STROKE_SPACING_FRACTION: f32 = 0.10;
    // Approximate pixels-per-meter at typical viewport distance. The
    // proper conversion needs the camera projection at the brush hit
    // depth, which isn't readily available editor-side; this
    // heuristic is close enough for normal sculpting cameras (1m of
    // world space at ~5m camera distance with 60° FOV on a 1080-tall
    // viewport projects to roughly 200 px). If the user sets a wildly
    // different zoom and stamps come out clustered or sparse, the
    // proper fix is to plumb the projected radius back from the
    // engine each frame; defer until we hit that case.
    const APPROX_PIXELS_PER_METER: f32 = 200.0;
    // Floor on the spacing in screen pixels — a tiny world-space
    // brush would otherwise compute a sub-pixel spacing and stamp
    // every frame. 4 px keeps mouse jitter from compounding even at
    // micro-brush sizes.
    const MIN_STROKE_SPACING_PX: f32 = 4.0;
    // Monotonic stroke counter. Incremented on every LMB-down that
    // starts a sculpt stroke; every stamp within the same stroke
    // ships the same value so the scene-manager can detect stroke
    // boundaries and clear its per-stroke "already touched" cell
    // set. Pre-increment means the first stroke uses 1; the scene
    // mgr's initial 0 sentinel never collides with a real stroke.
    let sculpt_stroke_seq = std::cell::Cell::new(0u64);

    let cmd_tx = cmd.0.clone();
    let surface_for_handler = surface.clone();
    surface.set_event_handler(move |event| {
        use SurfaceEvent::*;

        // Check if surface size changed — send Resize to engine.
        {
            let (w, h) = surface_for_handler.layout_size();
            let w = w.max(64);
            let h = h.max(64);
            // Only send resize occasionally — every mouse event is fine,
            // the engine no-ops if the size hasn't changed.
            let _ = cmd_tx.send(arvx_engine::EngineCommand::Resize {
                id: PANEL_VIEWPORT, width: w, height: h,
            });
        }

        // Small helper that reads the current paint signals out of the
        // store and sends a `PaintAtPixel` for the given pixel. The
        // engine coalesces picks (only one in-flight at a time) so
        // firing this on every MouseMove during a drag is safe —
        // intermediate requests get replaced, not queued.
        let send_paint_stamp = |x: f32, y: f32| {
            let _ = cmd_tx.send(arvx_engine::EngineCommand::PaintAtPixel {
                id: PANEL_VIEWPORT,
                x: x.max(0.0) as u32,
                y: y.max(0.0) as u32,
                radius: store.paint_radius.get(),
                color: store.paint_color.get(),
                strength: store.paint_strength.get(),
                falloff: store.paint_falloff.get(),
                mode: store.paint_mode.get(),
                material_id: store.selected_material.get().unwrap_or(0),
            });
        };

        // Sculpt equivalent — same single-in-flight coalescing as paint.
        let send_sculpt_stamp = |x: f32, y: f32| {
            // FalloffCurve is currently hardcoded to Smooth (Blender's
            // default Draw / Inflate curve). The legacy
            // `sculpt_falloff` slider isn't wired through yet — a
            // curve-shape picker UI replaces it later.
            let _ = cmd_tx.send(arvx_engine::EngineCommand::SculptAtPixel {
                id: PANEL_VIEWPORT,
                x: x.max(0.0) as u32,
                y: y.max(0.0) as u32,
                radius: store.sculpt_radius.get(),
                falloff_curve: arvx_engine::FalloffCurve::Smooth,
                strength: store.sculpt_strength.get(),
                stroke_seq: sculpt_stroke_seq.get(),
                mode: store.sculpt_mode.get(),
                material_id: store.selected_material.get().unwrap_or(0),
            });
        };

        match event {
            MouseMove { x, y } => {
                let dx = x - last_mx.get();
                let dy = y - last_my.get();
                last_mx.set(x);
                last_my.set(y);
                let _ = cmd_tx.send(arvx_engine::EngineCommand::MouseMove {
                    id: PANEL_VIEWPORT, x, y, dx, dy,
                });
                // Phase 9b: region drag-box — update the live rect's
                // far corner while LMB is held. The overlay redraws
                // reactively off the signal.
                if let Some((x0, y0, _, _)) = store.terrain_region_drag_rect.get() {
                    if lmb_held.get() {
                        store.terrain_region_drag_rect.set(Some((x0, y0, x, y)));
                    }
                }
                // Paint-drag: in paint mode with LMB held, each mouse
                // move issues another stamp. The engine's pick
                // in-flight gate naturally coalesces duplicates.
                if store.paint_active.get() && lmb_held.get() {
                    // Drag-paint: each mouse move issues another stamp
                    // while LMB is held. Hover tracking is GPU-driven
                    // (the brush-state probe pass reads gbuf at the
                    // engine-side `mouse_pos`), so no extra command.
                    send_paint_stamp(x, y);
                } else if store.sculpt_active.get() && lmb_held.get() {
                    // Stroke-spacing gate. Threshold scales with the
                    // brush's world radius so small brushes fire
                    // densely and big brushes fire sparsely — matches
                    // how Blender's `spacing` is a percentage of
                    // radius rather than a fixed pixel count.
                    let world_radius = store.sculpt_radius.get();
                    let spacing_px = (world_radius
                        * STROKE_SPACING_FRACTION
                        * APPROX_PIXELS_PER_METER)
                        .max(MIN_STROKE_SPACING_PX);
                    let dxs = x - last_sculpt_stamp_x.get();
                    let dys = y - last_sculpt_stamp_y.get();
                    if (dxs * dxs + dys * dys).sqrt() >= spacing_px {
                        send_sculpt_stamp(x, y);
                        last_sculpt_stamp_x.set(x);
                        last_sculpt_stamp_y.set(y);
                    }
                }
            }
            MouseDown { button, x, y } => {
                let _ = cmd_tx.send(arvx_engine::EngineCommand::MouseButton {
                    id: PANEL_VIEWPORT,
                    button: map_button(button),
                    pressed: true,
                });
                if button == SurfaceMouseButton::Left {
                    lmb_held.set(true);
                    if store.terrain_region_drag_armed.get() {
                        // Phase 9b: region drag-box mode owns LMB —
                        // start a drag rect at the cursor and suppress
                        // every other LMB-down side effect (pick,
                        // paint, sculpt). The rect updates on
                        // MouseMove; MouseUp commits it.
                        store.terrain_region_drag_rect.set(Some((x, y, x, y)));
                    } else if store.paint_active.get() {
                        // Paint mode owns LMB — suppress the normal
                        // click-select pick so the click doesn't
                        // deselect everything while painting.
                        send_paint_stamp(x, y);
                    } else if store.sculpt_active.get() {
                        // Sculpt owns LMB the same way paint does.
                        // Bump the stroke counter FIRST so this
                        // first stamp of the stroke ships the new
                        // value; the scene mgr sees a stroke
                        // transition and clears the per-stroke
                        // touched-cell set. Then fire the stamp and
                        // seed the spacing state at this cursor
                        // position so subsequent MouseMove samples
                        // are measured from here.
                        sculpt_stroke_seq.set(sculpt_stroke_seq.get().wrapping_add(1));
                        send_sculpt_stamp(x, y);
                        last_sculpt_stamp_x.set(x);
                        last_sculpt_stamp_y.set(y);
                    } else {
                        let _ = cmd_tx.send(arvx_engine::EngineCommand::Pick {
                            id: PANEL_VIEWPORT,
                            x: x as u32,
                            y: y as u32,
                        });
                    }
                }
            }
            MouseUp { button, .. } => {
                let _ = cmd_tx.send(arvx_engine::EngineCommand::MouseButton {
                    id: PANEL_VIEWPORT,
                    button: map_button(button),
                    pressed: false,
                });
                if button == SurfaceMouseButton::Left {
                    lmb_held.set(false);
                    // Phase 9b: commit the region drag-box if one is
                    // in flight. Disarms the toolbar so the next LMB
                    // returns to its normal pick / paint / sculpt
                    // behaviour — author re-clicks Region to draw
                    // another box.
                    if let Some((x0, y0, x1, y1)) = store.terrain_region_drag_rect.get() {
                        store.terrain_region_drag_rect.set(None);
                        store.terrain_region_drag_armed.set(false);
                        let _ = cmd_tx.send(
                            arvx_engine::EngineCommand::SetTerrainRegionFromScreenRect {
                                id: PANEL_VIEWPORT,
                                x0,
                                y0,
                                x1,
                                y1,
                            },
                        );
                    }
                }
            }
            MouseWheel { delta_y, .. } => {
                let _ = cmd_tx.send(arvx_engine::EngineCommand::Scroll {
                    id: PANEL_VIEWPORT, delta: delta_y,
                });
            }
            KeyDown(key_data) => {
                // Phase 9b: Escape cancels an in-progress region
                // drag-box and disarms the toolbar. Handled before
                // any other key wiring so it takes precedence.
                if key_data.code == "Escape" {
                    if store.terrain_region_drag_armed.get()
                        || store.terrain_region_drag_rect.get().is_some()
                    {
                        store.terrain_region_drag_armed.set(false);
                        store.terrain_region_drag_rect.set(None);
                    }
                }
                // Delete key → delete selected entity.
                if key_data.code == "Delete" || key_data.code == "Backspace" {
                    let _ = cmd_tx.send(arvx_engine::EngineCommand::DeleteSelected);
                }
                // F5 → toggle play mode.
                if key_data.code == "F5" {
                    if store.play_mode.get() {
                        let _ = cmd_tx.send(arvx_engine::EngineCommand::PlayStop);
                    } else {
                        let _ = cmd_tx.send(arvx_engine::EngineCommand::PlayStart);
                    }
                }
                // P → toggle paint mode. Same signal the PaintToolbar
                // button writes, so the overlay updates immediately,
                // and we mirror the SetPaintActive command so the
                // engine lights up the cursor wireframe.
                if key_data.code == "KeyP" {
                    let on = !store.paint_active.get();
                    store.paint_active.set(on);
                    let _ = cmd_tx.send(arvx_engine::EngineCommand::SetPaintActive {
                        active: on,
                        radius: store.paint_radius.get(),
                    });
                }
                if let Some(key) = map_key(&key_data.code) {
                    let _ = cmd_tx.send(arvx_engine::EngineCommand::KeyDown { key });
                }
            }
            KeyUp(key_data) => {
                if let Some(key) = map_key(&key_data.code) {
                    let _ = cmd_tx.send(arvx_engine::EngineCommand::KeyUp { key });
                }
            }
            Drop { .. } => {
                // Any of the three drag sources (model / generator /
                // preset) spawned a live preview on DragEnter and kept
                // it chasing the cursor. Commit retires the preview so
                // the entity stops tracking further picks and stays
                // where it last landed.
                let had_preview_drag = store.model_drag.get().is_some()
                    || store.generator_drag.get().is_some()
                    || store.generator_preset_drag.get().is_some();
                if had_preview_drag {
                    let _ = cmd_tx.send(arvx_engine::EngineCommand::DragPreviewCommit);
                    store.model_drag.set(None);
                    store.generator_drag.set(None);
                    store.generator_preset_drag.set(None);
                }
                // Material drag-and-drop: assign to selected entity.
                if let Some(mat_id) = store.material_drag.get() {
                    if let Some(entity_id) = store.selected_entity.get() {
                        let _ = cmd_tx.send(arvx_engine::EngineCommand::AssignMaterial {
                            entity_id,
                            material_id: mat_id,
                        });
                    }
                    store.material_drag.set(None);
                }
            }
            DragEnter { x, y } => {
                let px = (x as i32).max(0) as u32;
                let py = (y as i32).max(0) as u32;
                // Pick the first drag source that's active. Only one
                // of these can be set at a time (ondragstart sets one).
                let source = if let Some(path) = store.model_drag.get() {
                    Some(arvx_engine::DragPreviewSource::Asset { path })
                } else if let Some(name) = store.generator_drag.get() {
                    Some(arvx_engine::DragPreviewSource::Generator { name })
                } else if let Some(path) = store.generator_preset_drag.get() {
                    Some(arvx_engine::DragPreviewSource::GeneratorPreset { path })
                } else {
                    None
                };
                if let Some(source) = source {
                    // Hide rinch's built-in HTML drag ghost — the 3D
                    // preview in the viewport is the authoritative
                    // visual while the cursor is over this surface.
                    // Restored on DragLeave (which fires both when the
                    // cursor exits and after a successful Drop).
                    rinch::prelude::suppress_drag_ghost();
                    let _ = cmd_tx.send(arvx_engine::EngineCommand::DragPreviewEnter {
                        id: PANEL_VIEWPORT,
                        source,
                        x: px, y: py,
                    });
                }
            }
            DragOver { x, y } => {
                let any_active = store.model_drag.get().is_some()
                    || store.generator_drag.get().is_some()
                    || store.generator_preset_drag.get().is_some();
                if any_active {
                    let _ = cmd_tx.send(arvx_engine::EngineCommand::DragPreviewOver {
                        id: PANEL_VIEWPORT,
                        x: (x as i32).max(0) as u32,
                        y: (y as i32).max(0) as u32,
                    });
                }
            }
            DragLeave => {
                // `DragLeave` also fires after a successful `Drop`. The
                // Drop handler clears every drag signal, so on a valid
                // drop none of these checks pass and cancel is skipped.
                // A true drag-out-without-drop still has the signal set
                // and aborts the preview.
                let any_active = store.model_drag.get().is_some()
                    || store.generator_drag.get().is_some()
                    || store.generator_preset_drag.get().is_some();
                if any_active {
                    let _ = cmd_tx.send(arvx_engine::EngineCommand::DragPreviewCancel);
                }
                // Restore the HTML ghost either way (see DragEnter).
                rinch::prelude::restore_drag_ghost();
            }
            _ => {}
        }
    });

    rsx! {
        div {
            style: "display:flex;flex-direction:column;width:100%;height:100%;",
            // Fixed header — viewport-wide controls (play/stop, etc.)
            ViewportHeaderBar {}
            // Render area with floating gizmo overlay
            div {
                style: "flex:1;min-height:0;position:relative;",
                RenderSurface { surface: Some(surface.clone()) }
                EditModeToolbar {}
                BrushToolbar {}
                TerrainToolbar {}
                // Phase 9b: region drag-box overlay. Reactive on the
                // shared signal; visible only while a drag is in
                // progress.
                div {
                    style: {move || {
                        match store.terrain_region_drag_rect.get() {
                            Some((x0, y0, x1, y1)) => {
                                let lx = x0.min(x1);
                                let ly = y0.min(y1);
                                let w = (x1 - x0).abs();
                                let h = (y1 - y0).abs();
                                format!(
                                    "position:absolute;left:{lx}px;top:{ly}px;\
                                     width:{w}px;height:{h}px;\
                                     border:1.5px dashed #ffc107;\
                                     background:rgba(255,193,7,0.08);\
                                     pointer-events:none;z-index:25;"
                                )
                            }
                            None => "display:none;".to_string(),
                        }
                    }},
                }
            }
        }
    }
}
