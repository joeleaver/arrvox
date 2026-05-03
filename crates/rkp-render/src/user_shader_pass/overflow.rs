//! Async readback ring for the GPU overflow counters.
//!
//! Every `dispatch_regions` ends with a `copy_buffer_to_buffer` of the
//! GPU `overflow` storage buffer into one of three CPU staging slots.
//! After `queue.submit`, the dispatcher calls
//! [`OverflowReadback::advance`] to schedule the slot's `map_async`,
//! and [`OverflowReadback::drain_and_log`] to walk every slot, log any
//! that landed READY with a non-zero counter, and recycle them.
//!
//! 3 frames in flight matches typical wgpu queue depth, so readbacks
//! never stall the GPU. The dispatcher gates writes via
//! [`OverflowReadback::next_write_buffer`] (returns `None` if the
//! oldest slot is still PENDING / READY) so we don't double-book.
//!
//! Layout of the underlying GPU buffer matches the `OVERFLOW_*`
//! constants in `user_shader_geom.wgsl`:
//!   - `[0]`            = octree pool overflow
//!   - `[1]`            = brick pool overflow
//!   - `[2]`            = leaf-attr pool overflow
//!   - `[3]`            = fill queue overflow
//!   - `[4..4+MAX_DEPTH+1]` = active-queue overflow per BFS level

use super::dispatch::MAX_DEPTH;

/// Number of overflow counter slots in the GPU `overflow` buffer.
/// Layout (must match `OVERFLOW_*` constants in user_shader_geom.wgsl):
///   [0]    = octree pool overflow
///   [1]    = brick pool overflow
///   [2]    = leaf-attr pool overflow
///   [3]    = fill queue overflow
///   [4..4+MAX_DEPTH+1] = active-queue overflow per level
pub(super) const OVERFLOW_COUNTER_COUNT: usize = 4 + (MAX_DEPTH as usize + 1);
pub(super) const OVERFLOW_BUFFER_BYTES: u64 = OVERFLOW_COUNTER_COUNT as u64 * 4;

/// Number of frames we keep readback staging buffers in flight for the
/// overflow counters. 3 frames matches the typical wgpu queue depth so
/// readbacks don't stall the GPU; at any given moment we have one
/// "current" staging buffer being copied into and two pending
/// `map_async` results.
const OVERFLOW_READBACK_FRAMES: usize = 3;

const MAP_STATE_IDLE: u8 = 0;
const MAP_STATE_PENDING: u8 = 1;
const MAP_STATE_READY: u8 = 2;
const MAP_STATE_FAILED: u8 = 3;

/// CPU-side machinery for reading back the GPU overflow counters with
/// 3-frame ring buffering. We copy `overflow_buffer` into `slots[i]`
/// each frame, then `map_async` it. Frames in flight don't stall the
/// GPU; we drain the oldest mapped slot each frame and log if any
/// counter is non-zero.
pub(super) struct OverflowReadback {
    slots: [OverflowReadbackSlot; OVERFLOW_READBACK_FRAMES],
    next_write: usize,
    /// True if a slot's `map_async` has resolved and the buffer is
    /// ready to read. The flag is shared with the map callback via
    /// an `Arc<AtomicBool>`.
    map_states: [std::sync::Arc<std::sync::atomic::AtomicU8>; OVERFLOW_READBACK_FRAMES],
}

struct OverflowReadbackSlot {
    buffer: wgpu::Buffer,
    in_flight: bool,
}

impl OverflowReadback {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let make_slot = |i: usize| OverflowReadbackSlot {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("user_shader_geom overflow stage {i}")),
                size: OVERFLOW_BUFFER_BYTES,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            in_flight: false,
        };
        let make_state = || std::sync::Arc::new(
            std::sync::atomic::AtomicU8::new(MAP_STATE_IDLE),
        );
        Self {
            slots: [make_slot(0), make_slot(1), make_slot(2)],
            map_states: [make_state(), make_state(), make_state()],
            next_write: 0,
        }
    }

    /// Returns the staging buffer to copy into this frame, or `None`
    /// if the slot's previous map_async hasn't completed yet (we don't
    /// double-book).
    pub(super) fn next_write_buffer(&mut self) -> Option<&wgpu::Buffer> {
        let idx = self.next_write;
        let state = self.map_states[idx].load(std::sync::atomic::Ordering::Acquire);
        // IDLE → never used yet, free to write.
        // FAILED → previous map errored; we already reset the buffer
        //   in drain_and_log so it's free again.
        // PENDING → in flight; skip this frame to avoid clobbering.
        // READY → drain_and_log not yet called; skip too.
        if state == MAP_STATE_IDLE || state == MAP_STATE_FAILED {
            self.slots[idx].in_flight = true;
            Some(&self.slots[idx].buffer)
        } else {
            None
        }
    }

    /// Schedule map_async on the slot we just copied into. Call AFTER
    /// the queue.submit so the copy is in flight.
    pub(super) fn advance(&mut self) {
        let idx = self.next_write;
        if !self.slots[idx].in_flight {
            return;
        }
        self.slots[idx].in_flight = false;
        let state = std::sync::Arc::clone(&self.map_states[idx]);
        state.store(MAP_STATE_PENDING, std::sync::atomic::Ordering::Release);
        let buffer = &self.slots[idx].buffer;
        let slice = buffer.slice(0..OVERFLOW_BUFFER_BYTES);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let next = if result.is_ok() {
                MAP_STATE_READY
            } else {
                MAP_STATE_FAILED
            };
            state.store(next, std::sync::atomic::Ordering::Release);
        });
        self.next_write = (idx + 1) % OVERFLOW_READBACK_FRAMES;
    }

    /// Walk every slot; for each that's READY, read its bytes, log if
    /// non-zero, unmap, mark IDLE.
    pub(super) fn drain_and_log(&mut self) {
        for idx in 0..OVERFLOW_READBACK_FRAMES {
            let state = self.map_states[idx].load(std::sync::atomic::Ordering::Acquire);
            if state == MAP_STATE_READY {
                let buffer = &self.slots[idx].buffer;
                let slice = buffer.slice(0..OVERFLOW_BUFFER_BYTES);
                let counts: Vec<u32> = {
                    let view = slice.get_mapped_range();
                    bytemuck::cast_slice::<u8, u32>(&view).to_vec()
                };
                buffer.unmap();
                self.map_states[idx].store(MAP_STATE_IDLE, std::sync::atomic::Ordering::Release);
                if counts.iter().any(|c| *c != 0) {
                    log_overflow(&counts);
                }
            } else if state == MAP_STATE_FAILED {
                eprintln!("[user_shader_pass] overflow map_async failed in slot {idx}");
                self.map_states[idx].store(MAP_STATE_IDLE, std::sync::atomic::Ordering::Release);
            }
        }
    }
}

fn log_overflow(counts: &[u32]) {
    let octree = counts.first().copied().unwrap_or(0);
    let brick = counts.get(1).copied().unwrap_or(0);
    let leaf_attr = counts.get(2).copied().unwrap_or(0);
    let fill_queue = counts.get(3).copied().unwrap_or(0);
    eprintln!(
        "[user_shader_pass] OVERFLOW — octree:{octree} brick:{brick} \
         leaf_attr:{leaf_attr} fill_queue:{fill_queue}",
    );
    for level in 0..=(MAX_DEPTH as usize) {
        let c = counts.get(4 + level).copied().unwrap_or(0);
        if c != 0 {
            eprintln!("[user_shader_pass] OVERFLOW — active_queue[L={level}]:{c}");
        }
    }
}
