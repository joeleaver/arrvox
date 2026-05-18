//! Profiling history — engine-side ring buffer of per-frame timings.
//!
//! Each frame the engine pushes a [`FrameSample`] containing CPU phase
//! timings, GPU pass timings, and a few scene counters. The buffer is
//! capped at [`HISTORY_LEN`] frames; oldest entries are evicted.
//!
//! Two consumers:
//! - The editor reads the latest sample (via `StateUpdate.profiling`)
//!   and keeps its own ring of those for sparklines.
//! - MCP (when wired up) calls [`ProfilingHistory::stats`] for an
//!   aggregated summary that fits in a few hundred tokens, or
//!   [`ProfilingHistory::raw`] for individual samples.

use std::collections::VecDeque;

/// Frames retained for stats / raw snapshots. ~4 s at 60 fps.
pub const HISTORY_LEN: usize = 256;

/// Buckets in the downsampled total-frame-ms timeline returned by
/// [`ProfilingHistory::stats`]. Small enough to embed in an MCP reply.
pub const TIMELINE_BUCKETS: usize = 32;

/// Sim-thread CPU phase timings recorded each frame in
/// `ArvxEngine::submit_render_frame`. All values are milliseconds.
///
/// These are sim-thread only. The render thread owns wgpu and runs on
/// its own clock; its GPU work shows up in `gpu_passes` (also a sim-
/// time view: timestamps are produced GPU-side and shipped back via
/// `RenderResult`). For "real frame budget", compare CPU `total_ms`
/// (sim cap) against the sum of `gpu_passes` (GPU cap).
#[derive(Debug, Clone, Default)]
pub struct CpuPhaseTimings {
    /// Sim-side per-tick work: gameplay systems, physics, animation,
    /// camera/input, ECS scene-sync. Dominates a healthy frame.
    pub setup_ms: f32,
    /// Time spent assembling the [`RenderFrame`] snapshot (cloning
    /// gpu_objects, building per-VR camera/atmo/vol params, flattening
    /// procedural trees, etc.). Cheap if entity count is moderate.
    ///
    /// [`RenderFrame`]: crate::render_frame::RenderFrame
    pub snapshot_ms: f32,
    /// `inbox.submit()` plus end-of-tick housekeeping. Microseconds in
    /// the steady state — the inbox is single-slot and submit is a
    /// quick lock-and-store.
    pub submit_ms: f32,
    /// CPU frame total. Equals `setup + snapshot + submit`.
    pub total_ms: f32,
}

/// One frame's worth of profiling data.
#[derive(Debug, Clone)]
pub struct FrameSample {
    pub frame_idx: u64,
    pub cpu: CpuPhaseTimings,
    /// `(label, ms)` for each GPU pass that produced a timestamp this
    /// frame. Empty until wgpu_profiler has resolved its first frame
    /// (~3 frames of warmup); also empty for samples that landed
    /// before the render thread published a `RenderResult` for them.
    pub gpu_passes: Vec<(String, f32)>,
    pub gpu_object_count: u32,
    /// Wall-clock interval between consecutive render-thread
    /// iterations, in milliseconds. The "actual frame time" — what
    /// the editor surface sees as a frame rate. `0.0` until a
    /// `RenderResult` carrying this frame's render dt arrives back
    /// from the render thread (typically 1-2 frames after sim
    /// pushed the sample).
    pub render_dt_ms: f32,
}

/// One frame's data shipped over the StateUpdate snapshot. Same shape
/// as [`FrameSample`] minus the `Vec<String>` allocation cost — the
/// editor stores its own ring of these.
pub type ProfilingFrame = FrameSample;

/// Per-pass aggregated stats over a window. All values are
/// milliseconds; counts are sample populations.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PassStats {
    pub name: String,
    /// Number of frames in which this pass produced a sample.
    pub samples: u32,
    pub min: f32,
    pub mean: f32,
    pub p50: f32,
    pub p95: f32,
    pub p99: f32,
    pub max: f32,
}

/// Aggregated CPU-phase stats. Same five-number summary plus mean.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct PhaseStats {
    pub min: f32,
    pub mean: f32,
    pub p50: f32,
    pub p95: f32,
    pub p99: f32,
    pub max: f32,
}

/// MCP-shaped summary of the ring buffer. Compact: fits in a few
/// hundred tokens regardless of pass count.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ProfilingStats {
    pub frame_count: u32,
    pub window_ms: f32,
    pub cpu_total_ms: PhaseStats,
    pub cpu_setup_ms: PhaseStats,
    pub cpu_snapshot_ms: PhaseStats,
    pub cpu_submit_ms: PhaseStats,
    pub passes: Vec<PassStats>,
    /// `TIMELINE_BUCKETS` bucket means of `cpu.total_ms`, oldest first.
    /// When the window is smaller than the bucket count, trailing
    /// entries are zero.
    pub timeline_total_ms: Vec<f32>,
}

/// Ring buffer of [`FrameSample`]s.
pub struct ProfilingHistory {
    samples: VecDeque<FrameSample>,
    capacity: usize,
}

impl Default for ProfilingHistory {
    fn default() -> Self {
        Self::with_capacity(HISTORY_LEN)
    }
}

impl ProfilingHistory {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, sample: FrameSample) {
        if self.samples.len() == self.capacity {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn latest(&self) -> Option<&FrameSample> {
        self.samples.back()
    }

    /// Most recent sample whose render-thread data has been stitched
    /// in by [`Self::attach_render_data`]. Used by
    /// `build_state_update` for the editor-bound profiling payload —
    /// without this, the editor would always see the just-pushed
    /// frame whose GPU passes / render dt haven't arrived back from
    /// the render thread yet (typically 1-2 frames behind sim).
    pub fn latest_with_render_data(&self) -> Option<&FrameSample> {
        self.samples
            .iter()
            .rev()
            .find(|s| !s.gpu_passes.is_empty() || s.render_dt_ms > 0.0)
    }

    /// Attach late-arriving render-thread data to the matching frame.
    ///
    /// The render thread publishes per-pass GPU timings and its own
    /// observed iteration interval (`render_dt_ms`) on its own clock,
    /// typically 1-2 frames behind sim. This walks the ring (small,
    /// O(N) is fine) and updates the sample with the matching
    /// `frame_idx`. If the matching frame already evicted from the
    /// ring (very long render stall), the data is dropped — better
    /// than misattributing it to a different frame.
    ///
    /// `render_dt_ms` is `None` for the very first iteration (no
    /// prior interval to measure against) — in that case the sample's
    /// `render_dt_ms` stays at 0.
    pub fn attach_render_data(
        &mut self,
        frame_idx: u64,
        passes: Vec<(String, f32)>,
        render_dt_ms: Option<f32>,
    ) {
        if let Some(sample) = self.samples.iter_mut().rev().find(|s| s.frame_idx == frame_idx) {
            sample.gpu_passes = passes;
            if let Some(dt) = render_dt_ms {
                sample.render_dt_ms = dt;
            }
        }
    }

    /// Most-recent `n` samples, oldest first. Caps at the window size.
    pub fn raw(&self, n: usize) -> Vec<FrameSample> {
        let take = n.min(self.samples.len());
        let start = self.samples.len() - take;
        self.samples.iter().skip(start).cloned().collect()
    }

    /// Aggregate stats over the most-recent `window` frames (capped at
    /// the buffer size). Returns `None` if the buffer is empty.
    pub fn stats(&self, window: usize) -> Option<ProfilingStats> {
        if self.samples.is_empty() {
            return None;
        }
        let take = window.min(self.samples.len());
        let start = self.samples.len() - take;
        let slice: Vec<&FrameSample> = self.samples.iter().skip(start).collect();

        // CPU phases — pull each as a parallel f32 vec.
        let collect = |f: fn(&CpuPhaseTimings) -> f32| -> Vec<f32> {
            slice.iter().map(|s| f(&s.cpu)).collect()
        };
        let cpu_total = phase_stats(collect(|c| c.total_ms));
        let cpu_setup = phase_stats(collect(|c| c.setup_ms));
        let cpu_snapshot = phase_stats(collect(|c| c.snapshot_ms));
        let cpu_submit = phase_stats(collect(|c| c.submit_ms));

        // GPU passes — group by label, preserving insertion order from
        // the most-recent frame so the report reads in pipeline order.
        let mut order: Vec<String> = Vec::new();
        let mut buckets: std::collections::HashMap<String, Vec<f32>> =
            std::collections::HashMap::new();
        if let Some(latest) = slice.last() {
            for (label, _) in &latest.gpu_passes {
                if !buckets.contains_key(label) {
                    order.push(label.clone());
                    buckets.insert(label.clone(), Vec::new());
                }
            }
        }
        for s in &slice {
            for (label, ms) in &s.gpu_passes {
                buckets
                    .entry(label.clone())
                    .or_insert_with(|| {
                        order.push(label.clone());
                        Vec::new()
                    })
                    .push(*ms);
            }
        }
        let passes: Vec<PassStats> = order
            .into_iter()
            .map(|name| {
                let values = buckets.remove(&name).unwrap_or_default();
                let p = phase_stats(values.clone());
                PassStats {
                    name,
                    samples: values.len() as u32,
                    min: p.min,
                    mean: p.mean,
                    p50: p.p50,
                    p95: p.p95,
                    p99: p.p99,
                    max: p.max,
                }
            })
            .collect();

        // Timeline: bucket cpu.total_ms across TIMELINE_BUCKETS, oldest
        // first. Each bucket holds the mean of its frames.
        let mut timeline = vec![0.0f32; TIMELINE_BUCKETS];
        if !slice.is_empty() {
            for (i, sample) in slice.iter().enumerate() {
                let b = (i * TIMELINE_BUCKETS) / slice.len();
                timeline[b] += sample.cpu.total_ms;
            }
            // Normalize by per-bucket count.
            let mut counts = [0u32; TIMELINE_BUCKETS];
            for i in 0..slice.len() {
                let b = (i * TIMELINE_BUCKETS) / slice.len();
                counts[b] += 1;
            }
            for (v, c) in timeline.iter_mut().zip(counts.iter()) {
                if *c > 0 {
                    *v /= *c as f32;
                }
            }
        }

        let window_ms: f32 = slice.iter().map(|s| s.cpu.total_ms).sum();

        Some(ProfilingStats {
            frame_count: slice.len() as u32,
            window_ms,
            cpu_total_ms: cpu_total,
            cpu_setup_ms: cpu_setup,
            cpu_snapshot_ms: cpu_snapshot,
            cpu_submit_ms: cpu_submit,
            passes,
            timeline_total_ms: timeline,
        })
    }
}

fn phase_stats(mut values: Vec<f32>) -> PhaseStats {
    if values.is_empty() {
        return PhaseStats::default();
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    let pick = |q: f32| -> f32 {
        let idx = ((n - 1) as f32 * q).round() as usize;
        values[idx.min(n - 1)]
    };
    let sum: f32 = values.iter().sum();
    PhaseStats {
        min: values[0],
        mean: sum / n as f32,
        p50: pick(0.50),
        p95: pick(0.95),
        p99: pick(0.99),
        max: values[n - 1],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(idx: u64, total: f32, march: f32) -> FrameSample {
        FrameSample {
            frame_idx: idx,
            cpu: CpuPhaseTimings { total_ms: total, ..Default::default() },
            gpu_passes: vec![("march".to_string(), march)],
            gpu_object_count: 1,
            render_dt_ms: total,
        }
    }

    #[test]
    fn ring_buffer_evicts_oldest() {
        let mut h = ProfilingHistory::with_capacity(3);
        for i in 0..5 { h.push(sample(i, 10.0, 1.0)); }
        assert_eq!(h.len(), 3);
        assert_eq!(h.latest().unwrap().frame_idx, 4);
        let raw = h.raw(10);
        assert_eq!(raw[0].frame_idx, 2);
        assert_eq!(raw[2].frame_idx, 4);
    }

    #[test]
    fn stats_compute_percentiles() {
        let mut h = ProfilingHistory::with_capacity(100);
        for i in 0..100 {
            h.push(sample(i, i as f32, i as f32 * 0.1));
        }
        let s = h.stats(100).unwrap();
        assert_eq!(s.frame_count, 100);
        assert!((s.cpu_total_ms.min - 0.0).abs() < 1e-3);
        assert!((s.cpu_total_ms.max - 99.0).abs() < 1e-3);
        // p50 of 0..100 → ~49 or 50.
        assert!(s.cpu_total_ms.p50 >= 49.0 && s.cpu_total_ms.p50 <= 50.0);
        assert_eq!(s.passes.len(), 1);
        assert_eq!(s.passes[0].name, "march");
        assert!((s.passes[0].max - 9.9).abs() < 1e-3);
    }

    #[test]
    fn timeline_has_fixed_length() {
        let mut h = ProfilingHistory::with_capacity(64);
        for i in 0..64 { h.push(sample(i, 5.0, 1.0)); }
        let s = h.stats(64).unwrap();
        assert_eq!(s.timeline_total_ms.len(), TIMELINE_BUCKETS);
        for v in &s.timeline_total_ms {
            assert!((*v - 5.0).abs() < 1e-3);
        }
    }

    #[test]
    fn empty_returns_none() {
        let h = ProfilingHistory::default();
        assert!(h.stats(10).is_none());
        assert!(h.latest().is_none());
        assert!(h.raw(10).is_empty());
    }
}
