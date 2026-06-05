//! Sim-thread tick loop.
//!
//! `tick_loop` owns the engine's outer pacing + command-drain cycle:
//! build initial state, loop forever doing {drain commands, run
//! per-tick systems, submit a render frame, sleep to target pacing}
//! until `EngineCommand::Shutdown` returns `false` from
//! `process_command`. Called from `ArvxEngine::spawn` on the engine
//! thread.

use std::time::Instant;

use crossbeam::channel::Receiver;

use crate::command::EngineCommand;

use super::state::EngineState;
use super::{EngineConfig, FrameCallback, StateCallback};

pub(crate) fn tick_loop(
    cmd_rx: Receiver<EngineCommand>,
    frame_callback: FrameCallback,
    state_callback: StateCallback,
    config: EngineConfig,
) {
    // Hand the frame_callback to the render thread (constructed inside
    // EngineState::new). Sim no longer fires it directly — pixel
    // callbacks happen on the render thread after each VR's readback
    // drain. The callback closure is `Send`, so this just transfers
    // ownership across the thread boundary at spawn time. The
    // state_callback lives on EngineState so command handlers can
    // publish mid-handler progress snapshots (see `publish_phase`).
    let mut state = EngineState::new(&config, frame_callback, state_callback);
    state.console.info(format!("Engine started ({}x{})", config.width, config.height));

    // Try to load a pre-built gameplay dylib (if project is already set).
    // Normally the dylib is scaffolded + built when a project is opened.
    state.try_load_gameplay_dylib();

    let mut last_tick_start: Option<Instant> = None;

    loop {
        let frame_start = Instant::now();

        // Real wall-clock time since the previous tick started. Cap at 100ms
        // so a one-off hitch (e.g. an asset load) doesn't catapult dynamic
        // bodies; physics will fall behind real time during the hitch and
        // catch up on the next normal tick. First tick uses the target
        // pacing interval since there's no prior tick to measure against.
        let real_dt = match last_tick_start {
            Some(prev) => frame_start.duration_since(prev).as_secs_f32().min(0.1),
            None => 1.0 / 60.0,
        };
        last_tick_start = Some(frame_start);
        let inst_tick_hz = if real_dt > 0.0 { 1.0 / real_dt } else { 0.0 };
        state.tick_hz_ema = state.tick_hz_ema * 0.9 + inst_tick_hz * 0.1;

        // 1. Drain command queue.
        while let Ok(cmd) = cmd_rx.try_recv() {
            if !state.process_command(cmd) {
                eprintln!("[ArvxEngine] shutdown");
                return;
            }
        }

        // 1b. Process file watcher events + import events/completions + gameplay reload.
        state.process_file_events();
        state.pump_import_events();
        state.poll_import_completions();
        state.check_gameplay_reload();

        // 1b2. Integrate finished async bakes, then enqueue any new
        // work. Drain first so a bake that just completed gets applied
        // before its entity's `pending_bake` gets re-queued — avoids
        // an otherwise-harmless one-tick stale queue entry.
        state.drain_bake_results();
        state.update_dirty_procedurals();
        // Integrate a budget of deferred scene-load asset loads so a
        // heavy load reveals geometry progressively instead of freezing.
        state.drain_pending_asset_loads();

        // 1b3. Generator system — poll finished generator jobs,
        // detect param edits, submit stale jobs. Lives alongside the
        // bake pump because both flow through the same worker.
        state.tick_generators();

        // 1b4. Terrain streamer — materialise tiles around the camera,
        // integrate completed bakes, evict tiles outside the radius.
        // No-op when no Terrain is spawned.
        state.tick_terrain_streamer();

        // 1c. Step gameplay systems + physics if in play mode.
        //
        // Frame order: Update → flush → FixedUpdate → flush → Physics → LateUpdate → flush
        //
        // Gameplay runs before physics so scripts can set transforms on kinematic
        // bodies before physics reads them. Dynamic bodies have their transforms
        // overwritten by physics afterward (physics owns dynamic bodies).
        if state.play_state.is_none() {
            // Decay physics readout when not stepping so a stale 60 Hz doesn't
            // persist in the profiler after Stop.
            state.physics_hz_ema *= 0.9;
        }
        if state.play_state.is_some() {
            let dt = real_dt;
            /// Fixed-step duration for behavior FixedUpdate. Kept in
            /// sync with arvx-physics's default timestep on purpose —
            /// running them at the same rate means a FixedUpdate
            /// system that manipulates a kinematic body sees the
            /// physics world integrated at matching cadence.
            const FIXED_DT: f32 = 1.0 / 60.0;
            /// Cap on fixed steps per render frame. Mirrors physics'
            /// "spiral of death" guard — if we ever fall this far
            /// behind, drop the residual rather than try to catch up.
            const MAX_FIXED_STEPS: u32 = 8;

            state.play_total_time += dt as f64;
            state.play_frame_count += 1;

            // Phase 1: Update — runs once per render frame at real_dt.
            if let Some(ref mut executor) = state.behavior_executor {
                executor.tick_update(
                    &state.gameplay_systems,
                    &mut state.world,
                    &mut state.behavior_commands,
                    &mut state.game_store,
                    dt,
                    state.play_total_time,
                    state.play_frame_count,
                );
                state.gpu_objects_dirty.mark_all();
            }

            // Phase 2: FixedUpdate — accumulator-driven. Runs zero or
            // more times at exactly FIXED_DT each. A 60 Hz render has
            // exactly one step per frame in steady state; a 240 Hz
            // render has a step every ~4 frames; a heavy hitch (say
            // 100 ms) would fire up to MAX_FIXED_STEPS back-to-back
            // and then drop the rest.
            if let Some(ref mut executor) = state.behavior_executor {
                state.behavior_fixed_accumulator += dt;
                let mut steps = 0u32;
                while state.behavior_fixed_accumulator >= FIXED_DT
                    && steps < MAX_FIXED_STEPS
                {
                    executor.tick_fixed_update(
                        &state.gameplay_systems,
                        &mut state.world,
                        &mut state.behavior_commands,
                        &mut state.game_store,
                        FIXED_DT,
                        state.play_total_time,
                        state.play_frame_count,
                    );
                    state.behavior_fixed_accumulator -= FIXED_DT;
                    steps += 1;
                }
                if steps == MAX_FIXED_STEPS {
                    // Spiral-of-death guard: drop residual so we
                    // don't keep trying to catch up next frame.
                    state.behavior_fixed_accumulator = 0.0;
                }
            }

            // Physics step (between FixedUpdate and LateUpdate).
            // Physics has its own Rapier-side accumulator so passing
            // real_dt is correct regardless of render rate.
            if let Some(ref mut play) = state.play_state {
                if play.step(dt, &mut state.world) {
                    state.gpu_objects_dirty.mark_all();
                }
                let substeps = play.last_step_substeps() as f32;
                let inst_hz = if dt > 0.0 { substeps / dt } else { 0.0 };
                state.physics_hz_ema = state.physics_hz_ema * 0.9 + inst_hz * 0.1;
            }

            // Phase 3: LateUpdate — once per render frame at real_dt.
            if let Some(ref mut executor) = state.behavior_executor {
                executor.tick_late(
                    &state.gameplay_systems,
                    &mut state.world,
                    &mut state.behavior_commands,
                    &mut state.game_store,
                    dt,
                    state.play_total_time,
                    state.play_frame_count,
                );
            }

            // Drain viewport requests from behaviors (e.g. set_active_camera).
            // The executor only touches the ECS world; viewport state lives
            // on EngineState so we apply these after the phases complete.
            let requests = state.behavior_commands.take_viewport_requests();
            for req in requests {
                state.apply_viewport_request(req);
            }
        }

        // 1d. Advance skeletal animations on real wall-clock dt so
        // playback rate is independent of render rate. Runs every
        // frame in both edit and play modes so animated characters
        // preview correctly in the editor.
        if crate::animation::tick(&mut state.world, real_dt) {
            state.gpu_objects_dirty.mark_all();
        }

        // 2. Update input system + camera with real_dt — fly mode
        // uses dt to scale velocity, so anything other than real_dt
        // makes the camera move at the wrong speed when the render
        // rate diverges from 60 Hz.
        state.input_system.evaluate();
        state.camera_control.update(
            &state.input_system,
            real_dt,
            &mut state.camera.position,
            &mut state.camera.yaw,
            &mut state.camera.pitch,
        );
        // F10 — debug dump camera position + look direction. Sized for
        // copy-paste into a test fixture; tile key included so the
        // user can immediately point at which `bake_tile` call to
        // reproduce.
        if state.input_system.just_pressed("debug.dump_camera") {
            let p = state.camera.position;
            let look = crate::camera::look_dir(state.camera.yaw, state.camera.pitch);
            let tile_x = (p.x / arvx_terrain::TILE_SIZE_M).floor() as i32;
            let tile_y = (p.y / arvx_terrain::TILE_SIZE_M).floor() as i32;
            let tile_z = (p.z / arvx_terrain::TILE_SIZE_M).floor() as i32;
            let msg = format!(
                "camera pos=({:.2}, {:.2}, {:.2}) look=({:.3}, {:.3}, {:.3}) \
                 yaw={:.2}° pitch={:.2}° -> level-0 tile=({}, {}, {})",
                p.x, p.y, p.z, look.x, look.y, look.z,
                state.camera.yaw.to_degrees(), state.camera.pitch.to_degrees(),
                tile_x, tile_y, tile_z,
            );
            eprintln!("[debug.dump_camera] {msg}");
            state.console.info(msg);
        }
        state.sync_main_viewport_from_legacy_camera();

        // 3. Update gizmo hover + drag — MAIN targets the entity
        // transform, BUILD targets the selected procedural node.
        state.update_gizmo();
        state.update_procedural_gizmo();
        // BUILD left-click picking is handled by `process_pick_result`
        // reading the GPU `gbuf_pick` texture (written every frame by
        // `proc_raymarch.wgsl`). Ghost-pick priority is applied there
        // too — see the BUILD+Raymarch branch in process_pick_result.

        // 4. Build the render snapshot and submit to the render thread.
        //    Pixel `frame_callback` fires from the render thread after
        //    each VR's composite readback drains; sim no longer touches
        //    it directly.
        state.submit_render_frame();

        // 6. Push state to client.
        let frame_time = frame_start.elapsed();
        let update = state.build_state_update(frame_time);
        (state.state_callback)(&update);

        // 7. Clear per-frame input state for next tick.
        state.input_system.begin_frame();

        // 8. Sim-loop pacing — sleep the remainder of the configured
        // sim target interval. `Uncapped` skips the sleep entirely
        // and lets the sim loop run as fast as its work allows.
        // Render pacing is separate and handled on the render thread.
        if let Some(target) = config.sim_pacing.target_interval() {
            let elapsed = frame_start.elapsed();
            if elapsed < target {
                std::thread::sleep(target - elapsed);
            }
        }
    }
}

impl EngineState {
    /// Drain every [`RenderResult`] the render thread has published
    /// since the previous tick. Applies pick decoding, updates the
    /// smoothed-cloud-sun-atten target, and stitches GPU pass timings
    /// into the most-recent profiling sample.
    ///
    /// Called from the top of [`Self::render_frame`]; safe to call
    /// when the channel is empty (no-op).
    pub(crate) fn drain_render_results(&mut self) {
        // Take a Vec rather than reborrow `self.render_worker` inside
        // the loop — pick processing wants `&mut self`, which would
        // otherwise alias the channel borrow.
        let mut latest_atten: Option<f32> = None;
        // (frame_idx, gpu_passes, render_dt_ms) — kept together so a
        // single `attach_render_data` call updates both fields on the
        // matching ring entry.
        let mut latest_render_data: Option<(u64, Vec<(String, f32)>, Option<f32>)> = None;
        let mut pick_results: Vec<crate::render_frame::PickResult> = Vec::new();
        // EMA alpha for the render-FPS readout. 0.1 = ~25-tick
        // settling at 60 Hz; same time-constant the tick/physics
        // EMAs use, keeps the panel readouts feeling consistent.
        const RENDER_HZ_EMA_ALPHA: f32 = 0.1;
        while let Ok(result) = self.render_worker.outbox.try_recv() {
            if !result.cloud_sun_atten_raw.is_nan() {
                latest_atten = Some(result.cloud_sun_atten_raw);
            }
            // Track latest result that carries either GPU passes OR a
            // render dt — both are stitched onto the matching frame
            // sample. We always overwrite so the "latest" wins on
            // multi-result drains; correlation by frame_index keeps
            // attribution honest even if results arrive out of order
            // (which they shouldn't, but the API doesn't forbid it).
            if !result.gpu_passes.is_empty() || result.render_dt_ms.is_some() {
                latest_render_data = Some((
                    result.frame_index,
                    result.gpu_passes,
                    result.render_dt_ms,
                ));
            }
            if let Some(pr) = result.pick_result {
                pick_results.push(pr);
            }
            // Fold render thread's observed iteration interval into
            // the FPS EMA. Skip the first iteration (`None`) and
            // any zero/negative dt (paranoia — clock can rarely tie
            // on the same nanosecond).
            if let Some(dt_ms) = result.render_dt_ms {
                if dt_ms > 0.0 {
                    let inst_hz = 1000.0 / dt_ms;
                    self.render_hz_ema = self.render_hz_ema * (1.0 - RENDER_HZ_EMA_ALPHA)
                        + inst_hz * RENDER_HZ_EMA_ALPHA;
                }
            }
            // Same EMA treatment for the delivered-frame rate, fed
            // only when a pixel ship actually fired.
            if let Some(dt_ms) = result.delivered_dt_ms {
                if dt_ms > 0.0 {
                    let inst_hz = 1000.0 / dt_ms;
                    self.delivered_hz_ema = self.delivered_hz_ema * (1.0 - RENDER_HZ_EMA_ALPHA)
                        + inst_hz * RENDER_HZ_EMA_ALPHA;
                }
            }
        }
        if let Some(a) = latest_atten {
            self.last_cloud_sun_atten_raw = a;
        }
        if let Some((frame_idx, passes, dt)) = latest_render_data {
            self.profiling.attach_render_data(frame_idx, passes, dt);
        }
        for pr in pick_results {
            self.process_pick_result(pr);
        }
    }
}
