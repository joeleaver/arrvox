//! Decoupled composite-readback: a poll thread + a present thread (P2).
//!
//! ## Why two threads, not one
//!
//! Before P2 the render thread was the frame producer, the sole `device.poll`
//! caller, AND the pixel shipper — gating new frames on readback-slot
//! availability. That coupled presentation to GPU-queue depth and froze under
//! load. P2 split readback off the render thread, but the FIRST cut put the
//! blit (`frame_callback`) on the same thread that owns `device.poll` — so a
//! slow blit stalled the poller, `on_submitted_work_done` stopped firing,
//! `inflight_submits` pinned at the cap, and the render thread silently
//! pacing-skipped: a self-inflicted freeze. The surface is a dumb blit target;
//! not stalling behind the blit is OUR job.
//!
//! So readback is split across TWO threads with a hard rule between them:
//!
//! - **Poll thread** ([`run_readback_poll_thread`]) — the SOLE `device.poll`
//!   caller. It arms `map_async`, polls, drains completions, copies the newest
//!   frame per viewport into a shared [`PresentState`] slot, and recycles the
//!   ring slot. It drives every async map (composite + pick + sun-atten +
//!   lod-stats) and every `on_submitted_work_done` callback. **It NEVER calls
//!   `frame_callback` and NEVER blocks on the blit** — so the render thread's
//!   queue-depth pacing can always drain.
//!
//! - **Present thread** ([`run_present_thread`]) — owns the editor
//!   `frame_callback` (the surface blit). It waits for a freshly-published
//!   frame, moves it out of the shared slot, and blits it. Newest-wins: if the
//!   blit is slow, the poll thread keeps overwriting the slot with newer
//!   frames and the present thread picks up the newest when it's free; stale
//!   frames are simply dropped. A slow blit can ONLY drop frames — it can
//!   never stall the poller, the pacing, or the GPU pipeline.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crossbeam::channel::{Receiver, RecvTimeoutError, Sender};

use crate::viewport::ViewportId;

use super::state::{FrameCallback, MIN_FRAME_CALLBACK_INTERVAL};

/// A composite-readback job handed from the render thread to the poll thread
/// after the render thread copied the composite into `buffer` and submitted the
/// encoder. The poll thread owns the buffer's `map_async` → read → publish →
/// unmap → recycle lifecycle from here.
pub(super) struct ReadbackJob {
    pub vp_id: ViewportId,
    /// Ring slot index, echoed back in [`SlotFree`] so the render thread knows
    /// which slot became writable.
    pub slot: usize,
    /// Clone of the ring buffer the composite was copied into (same underlying
    /// GPU allocation as the render thread's `ReadbackRing.buffers[slot]` until
    /// a resize recreates it — see `generation`).
    pub buffer: wgpu::Buffer,
    /// Monotonic, engine-global. Drives newest-wins publish ordering AND the
    /// slot-free match (so a stale `SlotFree` for a buffer recreated by a
    /// resize can't free a newly-issued slot).
    pub generation: u64,
    pub width: u32,
    pub height: u32,
    pub padded_row: u32,
}

/// Signal from the poll thread back to the render thread: a ring slot's buffer
/// has been read + unmapped and is writable again. Carries the `generation`
/// that occupied the slot so the render thread ignores a stale free for a slot
/// that has since been re-issued (resize race).
pub(super) struct SlotFree {
    pub vp_id: ViewportId,
    pub slot: usize,
    pub generation: u64,
}

/// Render-side handles to the readback plumbing, stored on `RenderState`.
pub(super) struct RenderReadbackHandles {
    /// Render thread → poll thread: hand off a copied+submitted composite.
    pub job_tx: Sender<ReadbackJob>,
    /// Poll thread → render thread: a ring slot is writable again.
    pub slot_free_rx: Receiver<SlotFree>,
    /// Monotonic readback generation counter (shared so all threads agree).
    pub generation: Arc<AtomicU64>,
    /// Submitted-but-not-GPU-complete command buffers. The render thread bumps
    /// it on submit + registers an `on_submitted_work_done` decrement; it paces
    /// new frames against this instead of readback slots.
    pub inflight_submits: Arc<AtomicU32>,
    /// f32-bits of the wall-clock ms between the last two BLITTED frames,
    /// written by the present thread, read by the render thread when building
    /// `RenderResult` so the editor's delivered-FPS panel reflects real ship
    /// cadence. NaN bits = nothing blitted yet.
    pub delivered_dt_bits: Arc<AtomicU32>,
}

/// The latest decoded composite for one viewport, awaiting blit.
struct LatestFrame {
    generation: u64,
    pixels: Vec<u8>,
    width: u32,
    height: u32,
}

/// Newest-wins handoff from the poll thread to the present thread: one slot per
/// viewport, plus a shutdown flag. Guarded by a `Mutex` + woken by a `Condvar`.
pub(super) struct PresentState {
    frames: HashMap<ViewportId, LatestFrame>,
    shutdown: bool,
}

/// Shared present handoff (mutex + condvar).
pub(super) type PresentHandle = Arc<(Mutex<PresentState>, Condvar)>;

pub(super) fn new_present_handle() -> PresentHandle {
    Arc::new((
        Mutex::new(PresentState {
            frames: HashMap::new(),
            shutdown: false,
        }),
        Condvar::new(),
    ))
}

/// One composite buffer awaiting its `map_async` completion.
struct InFlight {
    job: ReadbackJob,
    rx: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
}

/// Poll-thread entry point — the SOLE `device.poll` caller. Returns when
/// `job_rx` disconnects (render thread dropped its `RenderState` = shutdown),
/// at which point it signals the present thread to shut down too.
pub(super) fn run_readback_poll_thread(
    device: wgpu::Device,
    job_rx: Receiver<ReadbackJob>,
    slot_free_tx: Sender<SlotFree>,
    present: PresentHandle,
) {
    let mut in_flight: Vec<InFlight> = Vec::new();

    loop {
        // ── 1. Pull handed-off jobs and arm their map_async. When nothing is
        //       in flight, block on the channel (modest timeout) so we don't
        //       busy-spin while idle; the timeout still lets us poll once for a
        //       stray pick/sun-atten map_async. ──
        if in_flight.is_empty() {
            match job_rx.recv_timeout(Duration::from_millis(8)) {
                Ok(job) => arm(&mut in_flight, job),
                Err(RecvTimeoutError::Timeout) => {
                    let _ = device.poll(wgpu::PollType::Poll);
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        while let Ok(job) = job_rx.try_recv() {
            arm(&mut in_flight, job);
        }

        // ── 2. Sole device.poll. Bounded wait — wake promptly on GPU
        //       completion, never hang. Drives ALL map_async callbacks
        //       (composite + pick + sun-atten + lod-stats) and ALL
        //       on_submitted_work_done callbacks (the render thread's pacing). ──
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
                // Map error — slot is free again but the buffer is NOT mapped,
                // so it must not be read or unmapped.
                Ok(Err(_)) => completed.push((f, false)),
                Err(std::sync::mpsc::TryRecvError::Empty) => still.push(f),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => completed.push((f, false)),
            }
        }
        in_flight = still;
        if completed.is_empty() {
            continue;
        }

        // ── 4. Newest-wins per viewport: PUBLISH the highest completed+mapped
        //       generation per viewport to the present slot (the present thread
        //       blits it). Never call frame_callback here. ──
        let mut newest_per_vp: HashMap<ViewportId, u64> = HashMap::new();
        for (f, mapped) in &completed {
            if *mapped {
                let e = newest_per_vp.entry(f.job.vp_id).or_insert(0);
                if f.job.generation > *e {
                    *e = f.job.generation;
                }
            }
        }
        for (f, mapped) in &completed {
            if !*mapped {
                continue;
            }
            let is_newest = newest_per_vp.get(&f.job.vp_id).copied() == Some(f.job.generation);
            if !is_newest {
                continue;
            }
            // Copy pixels OUT now so the buffer can be recycled immediately —
            // the blit (on the present thread) never holds the ring buffer.
            let slice = f.job.buffer.slice(..);
            let data = slice.get_mapped_range();
            let pixels = unpad(&data, f.job.width, f.job.height, f.job.padded_row);
            drop(data);
            publish(&present, f.job.vp_id, f.job.generation, pixels, f.job.width, f.job.height);
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

    // Render thread is gone → tell the present thread to exit.
    let (lock, cvar) = &*present;
    if let Ok(mut st) = lock.lock() {
        st.shutdown = true;
    }
    cvar.notify_all();
}

/// Present-thread entry point — owns the surface blit (`frame_callback`). Blits
/// the newest published frame per viewport, newest-wins + rate-limited. A slow
/// blit only drops frames; it can never stall the poll thread. Returns on
/// shutdown.
pub(super) fn run_present_thread(
    present: PresentHandle,
    frame_callback: FrameCallback,
    delivered_dt_bits: Arc<AtomicU32>,
) {
    let (lock, cvar) = &*present;
    let mut last_ship: Option<Instant> = None;

    loop {
        // Take whatever has been published (move it out — no copy under lock).
        let batch: Vec<(ViewportId, LatestFrame)> = {
            let mut st = match lock.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            while st.frames.is_empty() && !st.shutdown {
                st = match cvar.wait_timeout(st, Duration::from_millis(200)) {
                    Ok((g, _)) => g,
                    Err(_) => return,
                };
            }
            if st.shutdown && st.frames.is_empty() {
                return;
            }
            st.frames.drain().collect()
        };

        // Light rate-limit (protects the surface writer from being hammered
        // faster than it can composite). Newest-wins already coalesces; this
        // just caps the blit cadence. Applied OUTSIDE the lock so the poll
        // thread can keep publishing while we wait.
        if let Some(prev) = last_ship {
            let since = prev.elapsed();
            if since < MIN_FRAME_CALLBACK_INTERVAL {
                std::thread::sleep(MIN_FRAME_CALLBACK_INTERVAL - since);
            }
        }

        let now = Instant::now();
        for (vp, f) in batch {
            frame_callback(vp, &f.pixels, f.width, f.height);
        }
        if let Some(prev) = last_ship {
            let dt_ms = now.duration_since(prev).as_secs_f32() * 1000.0;
            delivered_dt_bits.store(dt_ms.to_bits(), Ordering::Relaxed);
        }
        last_ship = Some(now);
    }
}

/// Publish `pixels` for `vp` into the present slot if newer than what's there,
/// then wake the present thread. Holds the lock only for the move + insert.
fn publish(
    present: &PresentHandle,
    vp: ViewportId,
    generation: u64,
    pixels: Vec<u8>,
    width: u32,
    height: u32,
) {
    let (lock, cvar) = &**present;
    {
        let mut st = match lock.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        let newer = st
            .frames
            .get(&vp)
            .map(|f| f.generation < generation)
            .unwrap_or(true);
        if newer {
            st.frames.insert(
                vp,
                LatestFrame {
                    generation,
                    pixels,
                    width,
                    height,
                },
            );
        }
    }
    cvar.notify_one();
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
