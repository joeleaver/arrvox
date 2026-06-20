//! Dedicated composite-readback poll thread (P2 — decoupled readback present).
//!
//! ## Why this exists
//!
//! Before P2 the render thread was simultaneously the frame *producer*, the
//! sole `device.poll` caller, AND the pixel *shipper*: it rendered a frame,
//! copied the composite into a 3-slot CPU readback ring, polled the device to
//! drive the `map_async` callbacks, drained completed slots, and invoked the
//! editor [`FrameCallback`] — all inline. That coupled **presentation liveness
//! to readback-slot availability** which is coupled to GPU-queue depth: a cold
//! load burst inflated the queue, every readback slot stayed pending, and the
//! render thread either spun (stale surface) or hard-blocked on
//! `wait_indefinitely` (frozen renderer). No gate over a CPU ring can be made
//! robust — it can only pick stale or frozen.
//!
//! ## The split
//!
//! P2 moves *everything readback* onto this thread:
//!   - it is the **sole `device.poll` caller** (so it also services the pick +
//!     cloud-sun-atten + LOD-stats `map_async` callbacks, which fire on
//!     whichever thread polls — their results land in atomics / mpsc channels
//!     the render thread reads, so correctness is preserved);
//!   - it issues `map_async` on each composite buffer handed to it, waits for
//!     completion, copies the un-padded pixels out, and ships the **newest**
//!     frame per viewport to the [`FrameCallback`] (older completed frames are
//!     dropped — newest-wins; the editor only displays the freshest anyway);
//!   - it recycles the ring slot back to the render thread via [`SlotFree`]
//!     once the buffer is unmapped, so the render thread can write it again.
//!
//! The render thread now only: renders, copies the composite into a free ring
//! slot, submits, and hands a [`ReadbackJob`] over. It **never** blocks on
//! readback; queue depth is bounded instead by `on_submitted_work_done`
//! pacing (see `frame/encode.rs` + `loop_thread.rs`). There is no gate.
//!
//! `wgpu::Buffer` is a cheap `Clone` (Arc-backed) in wgpu 29, so the render
//! thread and this thread can both hold a handle to the same ring buffer; the
//! per-slot generation tag (see [`SlotFree`]) makes slot ownership unambiguous
//! even across a viewport resize that recreates the buffers underneath an
//! in-flight job.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam::channel::{Receiver, RecvTimeoutError, Sender};

use crate::viewport::ViewportId;

use super::state::{FrameCallback, MIN_FRAME_CALLBACK_INTERVAL};

/// A composite-readback job handed from the render thread to the poll thread
/// after the render thread has copied the composite into `buffer` and
/// submitted the encoder. The poll thread owns the buffer's `map_async` →
/// read → unmap → recycle lifecycle from here.
pub(super) struct ReadbackJob {
    pub vp_id: ViewportId,
    /// Ring slot index, echoed back in [`SlotFree`] so the render thread knows
    /// which slot became writable.
    pub slot: usize,
    /// Clone of the ring buffer the composite was copied into. Same underlying
    /// GPU allocation as the render thread's `ReadbackRing.buffers[slot]`
    /// (until a resize recreates it — see `generation`).
    pub buffer: wgpu::Buffer,
    /// Monotonic, engine-global. Drives BOTH newest-wins ship ordering AND the
    /// slot-free match (the render thread only frees a slot when the freed
    /// generation equals the one currently in flight for that slot, so a stale
    /// `SlotFree` for a buffer recreated by a resize can't free the new job).
    pub generation: u64,
    pub width: u32,
    pub height: u32,
    pub padded_row: u32,
}

/// Signal from the poll thread back to the render thread: a ring slot's buffer
/// has been read + unmapped and is writable again. Carries the `generation`
/// that occupied the slot so the render thread can ignore a stale free for a
/// slot that has since been re-issued (resize race).
pub(super) struct SlotFree {
    pub vp_id: ViewportId,
    pub slot: usize,
    pub generation: u64,
}

/// Render-side handles to the readback plumbing, stored on `RenderState`.
/// Created in `RenderWorker::spawn`; the matching receive/own ends live on the
/// poll thread.
pub(super) struct RenderReadbackHandles {
    /// Render thread → poll thread: hand off a copied+submitted composite.
    pub job_tx: Sender<ReadbackJob>,
    /// Poll thread → render thread: a ring slot is writable again.
    pub slot_free_rx: Receiver<SlotFree>,
    /// Monotonic readback generation counter (shared so both threads agree).
    pub generation: Arc<AtomicU64>,
    /// Count of submitted-but-not-GPU-complete command buffers. The render
    /// thread bumps it on submit + registers an `on_submitted_work_done`
    /// decrement; it paces new frames against this instead of readback slots.
    pub inflight_submits: Arc<AtomicU32>,
    /// f32-bits of the wall-clock ms between the last two shipped frames,
    /// written by the poll thread, read by the render thread for the
    /// "delivered FPS" panel value. NaN bits = nothing shipped yet.
    pub delivered_dt_bits: Arc<AtomicU32>,
}

/// One composite buffer awaiting its `map_async` completion.
struct InFlight {
    job: ReadbackJob,
    rx: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
}

/// Poll-thread entry point. Returns when `job_rx` disconnects (the render
/// thread dropped its `RenderState`, i.e. shutdown).
pub(super) fn run_readback_poll_thread(
    device: wgpu::Device,
    job_rx: Receiver<ReadbackJob>,
    slot_free_tx: Sender<SlotFree>,
    frame_callback: FrameCallback,
    delivered_dt_bits: Arc<AtomicU32>,
) {
    let mut in_flight: Vec<InFlight> = Vec::new();
    // Highest generation already shipped per viewport — guards against
    // re-shipping a stale frame after a newer one already went out.
    let mut last_shipped_gen: HashMap<ViewportId, u64> = HashMap::new();
    // Single ship timestamp across all viewports, mirroring the render-thread
    // rate-limit this replaces (protects rinch's surface-buffer Mutex).
    let mut last_ship: Option<Instant> = None;

    loop {
        // ── 1. Pull any handed-off jobs and arm their map_async. ──
        // When nothing is in flight, block on the channel (with a modest
        // timeout) so we don't busy-spin while idle; the timeout still lets us
        // do one cheap poll to service a stray pick/sun-atten map_async.
        if in_flight.is_empty() {
            match job_rx.recv_timeout(Duration::from_millis(8)) {
                Ok(job) => arm(&mut in_flight, job),
                Err(RecvTimeoutError::Timeout) => {
                    // Idle: drive callbacks once for picks/sun-atten/inflight
                    // decrements, then loop back to block again.
                    let _ = device.poll(wgpu::PollType::Poll);
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => return,
            }
        }
        while let Ok(job) = job_rx.try_recv() {
            arm(&mut in_flight, job);
        }

        // ── 2. Sole device.poll. Bounded wait so we wake promptly when GPU
        //       work completes but never hang. Drives ALL map_async callbacks
        //       on this device (composite + pick + sun-atten + LOD stats) and
        //       all on_submitted_work_done callbacks (inflight pacing). ──
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_millis(4)),
        });

        // ── 3. Partition in-flight into completed / still-pending. ──
        let mut completed: Vec<(InFlight, bool)> = Vec::new(); // (job, mapped_ok)
        let mut still: Vec<InFlight> = Vec::new();
        for f in in_flight.drain(..) {
            match f.rx.try_recv() {
                Ok(Ok(())) => completed.push((f, true)),
                // Map error (device lost, etc.) — slot is free again but the
                // buffer is NOT mapped, so it must not be read or unmapped.
                Ok(Err(_)) => completed.push((f, false)),
                Err(std::sync::mpsc::TryRecvError::Empty) => still.push(f),
                // Callback sender dropped without firing — treat as a failed,
                // unmapped completion so the slot recycles.
                Err(std::sync::mpsc::TryRecvError::Disconnected) => completed.push((f, false)),
            }
        }
        in_flight = still;
        if completed.is_empty() {
            continue;
        }

        // ── 4. Newest-wins ship. Find the highest completed+mapped generation
        //       per viewport; ship only that one (older completed frames are
        //       dropped). A single rate-limit gates the whole cycle. ──
        let mut newest_per_vp: HashMap<ViewportId, u64> = HashMap::new();
        for (f, mapped) in &completed {
            if *mapped {
                let e = newest_per_vp.entry(f.job.vp_id).or_insert(0);
                if f.job.generation > *e {
                    *e = f.job.generation;
                }
            }
        }
        let now = Instant::now();
        let time_ok = last_ship
            .map(|t| now.duration_since(t) >= MIN_FRAME_CALLBACK_INTERVAL)
            .unwrap_or(true);
        let mut shipped_any = false;
        if time_ok {
            for (f, mapped) in &completed {
                if !*mapped {
                    continue;
                }
                let is_newest = newest_per_vp.get(&f.job.vp_id).copied() == Some(f.job.generation);
                let gen_ok = last_shipped_gen
                    .get(&f.job.vp_id)
                    .copied()
                    .unwrap_or(0)
                    < f.job.generation;
                if is_newest && gen_ok {
                    let slice = f.job.buffer.slice(..);
                    let data = slice.get_mapped_range();
                    let pixels = unpad(&data, f.job.width, f.job.height, f.job.padded_row);
                    drop(data);
                    frame_callback(f.job.vp_id, &pixels, f.job.width, f.job.height);
                    last_shipped_gen.insert(f.job.vp_id, f.job.generation);
                    shipped_any = true;
                }
            }
        }
        if shipped_any {
            if let Some(prev) = last_ship {
                let dt_ms = now.duration_since(prev).as_secs_f32() * 1000.0;
                delivered_dt_bits.store(dt_ms.to_bits(), Ordering::Relaxed);
            }
            last_ship = Some(now);
        }

        // ── 5. Unmap (only mapped ones) + recycle EVERY completed slot. ──
        for (f, mapped) in completed {
            if mapped {
                f.job.buffer.unmap();
            }
            let _ = slot_free_tx.send(SlotFree {
                vp_id: f.job.vp_id,
                slot: f.job.slot,
                generation: f.job.generation,
            });
        }
    }
}

/// Arm `map_async` on a freshly-handed buffer and track it as in-flight.
fn arm(in_flight: &mut Vec<InFlight>, job: ReadbackJob) {
    let (tx, rx) = std::sync::mpsc::channel();
    job.buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
    in_flight.push(InFlight { job, rx });
}

/// Copy a row-padded mapped readback into a tightly-packed RGBA8 buffer.
/// (`copy_texture_to_buffer` pads each row to a 256-byte multiple.)
fn unpad(data: &[u8], width: u32, height: u32, padded_row: u32) -> Vec<u8> {
    let row = width as usize * 4;
    let mut out = vec![0u8; row * height as usize];
    for y in 0..height as usize {
        let src = y * padded_row as usize;
        let dst = y * row;
        if src + row <= data.len() && dst + row <= out.len() {
            out[dst..dst + row].copy_from_slice(&data[src..src + row]);
        }
    }
    out
}
