//! Profiling panel — live CPU + GPU per-frame timings.
//!
//! Architecture note: this panel prioritises render cost over code
//! brevity. Every row's DOM is built once at mount; per-tick updates
//! flow through `{|| …}` closures that only mutate text and style
//! attributes. Nothing in the hot path tears down or re-keys DOM.
//!
//! - CPU phase rows: six fixed rows, labels hard-coded, values read
//!   from the smoothed `store.profiling` signal via per-row closures.
//! - GPU pass rows: `for label in store.gpu_pass_labels.get() { … }`.
//!   The labels signal only changes when the pass set changes
//!   (effectively once, at startup) so the `for` doesn't churn.
//!   Per-row ms + bar width read from `store.profiling` separately.
//! - Sparkline: 128 slots via `for i in 0usize..128`. Static DOM; each
//!   slot's style closure reads its history slot plus a shared
//!   `max_ms` Memo so the reactive graph fans out cleanly.
//!
//! Data sent to `store.profiling` is smoothed (EMA α = 0.15) and
//! throttled to ≤60 Hz in the state callback (see main.rs). The
//! sparkline history stays raw.

use std::sync::Arc;

use rinch::prelude::*;

use crate::ui::store::{EditorStore, ProfilingWindow};

const SECTION_LABEL: &str = "font-size:10px;text-transform:uppercase;letter-spacing:0.06em;\
                             color:#7a7a7a;margin:10px 0 4px 0;";
const STAT_ROW: &str = "display:flex;align-items:center;gap:6px;min-height:18px;\
                        font-family:monospace;font-size:11px;color:#cfcfcf;";
const STAT_LABEL: &str = "width:90px;flex-shrink:0;color:#999;\
                          overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
const STAT_VALUE: &str = "width:64px;text-align:right;flex-shrink:0;color:#ddd;";
const BAR_TRACK: &str = "flex:1;min-width:40px;height:8px;background:#1e1e1e;\
                         border:1px solid #2e2e2e;border-radius:2px;overflow:hidden;\
                         position:relative;";
const TOTAL_ROW: &str = "display:flex;align-items:center;gap:6px;min-height:18px;\
                         font-family:monospace;font-size:11px;color:#fff;\
                         border-top:1px solid #2e2e2e;margin-top:2px;padding-top:4px;";
const BAR_FILL_BASE: &str = "position:absolute;left:0;top:0;bottom:0;";

const CPU_COLOR: &str = "#4fc3f7";
const GPU_COLOR: &str = "#ffb74d";

/// How many CPU phase bars (fixed set of labels below).
const CPU_PHASE_COUNT: usize = 3;
const CPU_PHASE_LABELS: [&str; CPU_PHASE_COUNT] = ["Setup", "Snapshot", "Submit"];

/// Pull the ms for a phase by index from the smoothed window. Kept
/// near the constants so the order stays aligned with the labels.
fn cpu_phase_ms(w: &ProfilingWindow, idx: usize) -> f32 {
    let c = &w.latest_cpu;
    match idx {
        0 => c.setup_ms,
        1 => c.snapshot_ms,
        2 => c.submit_ms,
        _ => 0.0,
    }
}

/// Green under 16.7 ms (60 fps), yellow to ~33 ms (30 fps), red beyond.
fn bar_color_for_ms(ms: f32) -> &'static str {
    if ms <= 16.7 { "#66bb6a" }
    else if ms <= 33.4 { "#ffa726" }
    else { "#ef5350" }
}

#[component]
pub fn ProfilingPanel() -> NodeHandle {
    let store = use_context::<EditorStore>();

    // Shared per-tick derivations — Memos dedupe by PartialEq so the
    // downstream closures fan out from a single read.
    let prof: Memo<Option<Arc<ProfilingWindow>>> = Memo::new(move || store.profiling.get());
    let cpu_total: Memo<f32> = Memo::new(move || {
        prof.get().map(|w| w.latest_cpu.total_ms).unwrap_or(0.0)
    });
    let gpu_total: Memo<f32> = Memo::new(move || {
        prof.get()
            .map(|w| w.latest_gpu.iter().map(|(_, m)| *m).sum::<f32>())
            .unwrap_or(0.0)
    });
    let gpu_max: Memo<f32> = Memo::new(move || {
        prof.get()
            .map(|w| {
                w.latest_gpu
                    .iter()
                    .map(|(_, m)| *m)
                    .fold(0.0f32, f32::max)
                    .max(1e-3)
            })
            .unwrap_or(1e-3)
    });
    let sparkline_max: Memo<f32> = Memo::new(move || {
        prof.get()
            .map(|w| {
                w.history
                    .iter()
                    .map(|(_, m)| *m)
                    .fold(0.0f32, f32::max)
                    .max(1e-3)
            })
            .unwrap_or(1e-3)
    });

    rsx! {
        div {
            style: "padding:8px 12px;color:#ccc;font-size:12px;overflow-y:auto;height:100%;",

            // ── Top-line rates ─────────────────────────────────
            //
            // After the sim/render thread split there are three rates
            // worth tracking; this section reads them in order of
            // user-visible-ness:
            //
            // - **Render FPS** — the render thread's actual iteration
            //   rate, EMA-smoothed. This is what the editor surface
            //   sees as a frame rate. Capped by `render_pacing` and
            //   limited by GPU capacity / sim availability.
            // - **Sim Hz** — sim tick rate (after `sim_pacing` sleep).
            //   Drives physics, animation, snapshot construction.
            //   Independent of render rate.
            // - **Sim CPU max / GPU max** — headroom indicators. They
            //   say "if we removed the cap this side could push N Hz."
            //   When both exceed Render FPS you're pace-limited; when
            //   either falls below, that side is the bottleneck.
            div { style: STAT_ROW,
                span { style: STAT_LABEL, "Render FPS" }
                span { style: STAT_VALUE, {move || format!("{:.0}", store.fps.get())} }
                span { style: "color:#777;font-size:10px;",
                    {move || {
                        let f = store.fps.get();
                        if f > 0.1 { format!("{:.2} ms", 1000.0 / f) } else { "—".into() }
                    }} }
            }
            div { style: STAT_ROW,
                span { style: STAT_LABEL, "Sim Hz" }
                span { style: STAT_VALUE, {move || format!("{:.1}", store.tick_hz.get())} }
            }
            div { style: STAT_ROW,
                span { style: STAT_LABEL, "Sim CPU max" }
                span { style: STAT_VALUE, {move || {
                    let c = cpu_total.get();
                    if c > 1e-3 { format!("{:.0}", 1000.0 / c) } else { "—".into() }
                }} }
                span { style: "color:#777;font-size:10px;",
                    {move || format!("{:.2} ms", cpu_total.get())} }
            }
            div { style: STAT_ROW,
                span { style: STAT_LABEL, "GPU max" }
                span { style: STAT_VALUE, {move || {
                    let g = gpu_total.get();
                    if g > 1e-3 { format!("{:.0}", 1000.0 / g) } else { "—".into() }
                }} }
                span { style: "color:#777;font-size:10px;",
                    {move || format!("{:.2} ms", gpu_total.get())} }
            }
            div { style: STAT_ROW,
                span { style: STAT_LABEL, "Physics Hz" }
                span { style: STAT_VALUE, {move || format!("{:.1}", store.physics_hz.get())} }
            }
            div { style: STAT_ROW,
                span { style: STAT_LABEL, "Objects" }
                span { style: STAT_VALUE, {move || format!("{}", store.gpu_object_count.get())} }
            }

            // ── Frame-time sparkline ───────────────────────────
            div { style: SECTION_LABEL, "Frame Time (last 128)" }
            div {
                style: "display:flex;align-items:flex-end;gap:1px;height:48px;\
                        background:#1a1a1a;border:1px solid #2e2e2e;border-radius:3px;\
                        padding:3px 4px;margin-bottom:6px;",
                // Static 128-slot DOM; each slot's style closure reads
                // its history entry. Slot 0 = oldest frame (leftmost),
                // slot 127 = newest (rightmost).
                for i in 0usize..128usize {
                    div {
                        key: i,
                        style: {move || {
                            let Some(window) = prof.get() else {
                                return String::from("display:none;");
                            };
                            let max = sparkline_max.get();
                            // History fills from the left as it
                            // accumulates; align-bottom + flex-end
                            // keep bars flush to the baseline.
                            let ms = window.history.get(i).map(|(_, m)| *m);
                            match ms {
                                Some(ms) => {
                                    let h_pct = (ms / max * 100.0).clamp(0.0, 100.0);
                                    let color = bar_color_for_ms(ms);
                                    format!(
                                        "flex:1;min-width:1px;max-width:4px;height:{h_pct:.1}%;\
                                         background:{color};border-radius:1px;"
                                    )
                                }
                                None => {
                                    // Pre-fill slots render as a faint
                                    // zero-height placeholder so flex
                                    // sizing stays stable as history
                                    // fills in.
                                    String::from(
                                        "flex:1;min-width:1px;max-width:4px;height:0%;",
                                    )
                                }
                            }
                        }},
                    }
                }
            }
            div { style: STAT_ROW,
                span { style: STAT_LABEL, "Latest frame" }
                span { style: STAT_VALUE, {move || {
                    // Latest entry in the sparkline history is the most
                    // recent measured render dt (actual frame time).
                    let dt = prof.get()
                        .and_then(|w| w.history.last().map(|(_, ms)| *ms))
                        .unwrap_or(0.0);
                    format!("{dt:.2} ms")
                }} }
                span {
                    style: "color:#777;font-size:10px;",
                    {move || format!("max {:.2}", sparkline_max.get())}
                }
            }

            // ── CPU phase breakdown ────────────────────────────
            div { style: SECTION_LABEL, "CPU Phases — Latest Frame" }
            for idx in 0..CPU_PHASE_COUNT {
                div {
                    key: idx,
                    style: STAT_ROW,
                    span { style: STAT_LABEL, {CPU_PHASE_LABELS[idx]} }
                    span { style: STAT_VALUE, {move || {
                        let ms = prof.get().map(|w| cpu_phase_ms(&w, idx)).unwrap_or(0.0);
                        format!("{ms:.2} ms")
                    }} }
                    div { style: BAR_TRACK,
                        div {
                            style: {move || {
                                let Some(window) = prof.get() else {
                                    return String::from(BAR_FILL_BASE);
                                };
                                let ms = cpu_phase_ms(&window, idx);
                                let denom = window.latest_cpu.total_ms.max(1e-3);
                                let pct = (ms / denom * 100.0).clamp(0.0, 100.0);
                                format!("{BAR_FILL_BASE}width:{pct:.1}%;background:{CPU_COLOR};")
                            }},
                        }
                    }
                }
            }
            div { style: TOTAL_ROW,
                span { style: STAT_LABEL, "Total" }
                span { style: STAT_VALUE, {move || format!("{:.2} ms", cpu_total.get())} }
                div { style: "flex:1;" }
            }

            // ── GPU pass breakdown ─────────────────────────────
            div { style: SECTION_LABEL, "GPU Passes — Latest Frame" }
            // The `for` source reads `gpu_pass_labels` which only
            // changes when the pass set changes. Per-row ms values
            // come in through reactive closures reading the main
            // profiling signal — so the for-loop doesn't churn DOM
            // per tick, only attribute updates flow through.
            // Iterate by index — `idx: usize` is Copy so multiple
            // `move ||` closures can each capture it without fighting
            // over a single String. Each closure looks up its label
            // and ms from the signals at render time.
            //
            // Why not capture the label String directly? The rsx macro
            // emits the child-text closure expression twice (once for
            // the initial text, once inside a create_effect), so a
            // non-Copy capture would move-into-two-places.
            for (idx, label) in store.gpu_pass_labels.get().as_ref().iter().cloned().enumerate().collect::<Vec<_>>() {
                // `idx` is captured by rsx-macro-generated closures below;
                // this explicit no-op read silences the false-positive
                // unused_variable warning that fires at the loop binding.
                let _ = idx;
                div {
                    key: label.clone(),
                    style: STAT_ROW,
                    span { style: STAT_LABEL, {label.clone()} }
                    span { style: STAT_VALUE, {move || {
                        let ms = prof.get()
                            .and_then(|w| w.latest_gpu.get(idx).map(|(_, m)| *m))
                            .unwrap_or(0.0);
                        format!("{ms:.2} ms")
                    }} }
                    div { style: BAR_TRACK,
                        div {
                            style: {move || {
                                let ms = prof.get()
                                    .and_then(|w| w.latest_gpu.get(idx).map(|(_, m)| *m))
                                    .unwrap_or(0.0);
                                let max = gpu_max.get();
                                let pct = (ms / max * 100.0).clamp(0.0, 100.0);
                                format!("{BAR_FILL_BASE}width:{pct:.1}%;background:{GPU_COLOR};")
                            }},
                        }
                    }
                }
            }
            div { style: TOTAL_ROW,
                span { style: STAT_LABEL, "GPU total" }
                span { style: STAT_VALUE, {move || format!("{:.2} ms", gpu_total.get())} }
                div { style: "flex:1;" }
            }
        }
    }
}
