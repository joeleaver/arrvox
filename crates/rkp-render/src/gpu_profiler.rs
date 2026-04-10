//! GPU timestamp profiler — measures per-pass GPU execution time.
//!
//! Creates a wgpu QuerySet with timestamp queries. Each pass writes a
//! begin/end timestamp. After submit, the resolved timestamps are read
//! back and the deltas printed.

/// Maximum number of timed passes (each uses 2 queries: begin + end).
const MAX_PASSES: u32 = 8;
const MAX_QUERIES: u32 = MAX_PASSES * 2;

pub struct GpuProfiler {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    /// Nanoseconds per timestamp tick (from adapter).
    timestamp_period: f32,
    /// Names for each pass slot.
    pass_names: Vec<&'static str>,
    /// Whether profiling is active.
    enabled: bool,
    /// Whether this frame had active dispatches.
    active: bool,
    /// Frame counter for periodic logging.
    frame_count: u64,
}

impl GpuProfiler {
    pub fn new(device: &wgpu::Device, timestamp_period: f32) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gpu_profiler"),
            ty: wgpu::QueryType::Timestamp,
            count: MAX_QUERIES,
        });

        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_profiler resolve"),
            size: (MAX_QUERIES as u64) * 8, // u64 per timestamp
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_profiler readback"),
            size: (MAX_QUERIES as u64) * 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            query_set,
            resolve_buffer,
            readback_buffer,
            timestamp_period,
            pass_names: Vec::new(),
            enabled: timestamp_period > 0.0,
            active: false,
            frame_count: 0,
        }
    }

    /// Register a pass name and return its slot index.
    /// Call during setup, not per frame.
    pub fn register_pass(&mut self, name: &'static str) -> u32 {
        let idx = self.pass_names.len() as u32;
        self.pass_names.push(name);
        idx
    }

    /// Get the ComputePassTimestampWrites for a pass slot.
    pub fn compute_timestamps(&self, slot: u32) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        if !self.enabled { return None; }
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(slot * 2),
            end_of_pass_write_index: Some(slot * 2 + 1),
        })
    }

    /// Resolve timestamps and copy to readback buffer.
    /// Call after all passes are recorded, before submit.
    /// `active` should be true only if compute passes actually dispatched.
    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder, active: bool) {
        self.active = active;
        if !self.enabled || !active { return; }
        let count = (self.pass_names.len() as u32 * 2).min(MAX_QUERIES);
        encoder.resolve_query_set(&self.query_set, 0..count, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer, 0,
            &self.readback_buffer, 0,
            count as u64 * 8,
        );
    }

    /// Read back timestamps and log results. Call AFTER device.poll() has completed
    /// (e.g., after the frame readback poll). Logs every `interval` frames.
    pub fn read_and_log(&mut self, device: &wgpu::Device, interval: u64) {
        self.frame_count += 1;
        // Skip first 2 frames (resolve buffer not yet written) and non-interval frames.
        if !self.enabled || !self.active || self.frame_count < 3 || self.frame_count % interval != 0 { return; }

        let count = self.pass_names.len();
        if count == 0 { return; }

        // Map the readback buffer. Don't block — if not ready, skip.
        let slice = self.readback_buffer.slice(..);
        let mapped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mapped_clone = mapped.clone();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            if r.is_ok() { mapped_clone.store(true, std::sync::atomic::Ordering::SeqCst); }
        });
        // Non-blocking poll — just process pending callbacks.
        let _ = device.poll(wgpu::PollType::Poll);

        if !mapped.load(std::sync::atomic::Ordering::SeqCst) {
            return; // not ready — try next interval
        }

        let data = slice.get_mapped_range();
        let timestamps: &[u64] = bytemuck::cast_slice(&data);

        let mut msg = String::from("[gpu]");
        let mut total_ns = 0u64;
        for i in 0..count {
            let begin = timestamps[i * 2];
            let end = timestamps[i * 2 + 1];
            let delta_ns = if end > begin { end - begin } else { 0 };
            let delta_ms = delta_ns as f64 * self.timestamp_period as f64 / 1_000_000.0;
            msg.push_str(&format!(" {}={:.2}ms", self.pass_names[i], delta_ms));
            total_ns += delta_ns;
        }
        let total_ms = total_ns as f64 * self.timestamp_period as f64 / 1_000_000.0;
        msg.push_str(&format!(" total={:.2}ms", total_ms));
        eprintln!("{msg}");

        drop(data);
        self.readback_buffer.unmap();
    }
}
