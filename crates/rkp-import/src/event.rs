//! Structured progress events for the import pipeline.
//!
//! The importer streams [`ImportEvent`]s through a
//! [`ProgressReporter`] so UIs can render live progress bars /
//! per-stage status without parsing stderr. Replaces the old
//! "print to stderr, hope the UI scrapes it" model.
//!
//! A default no-op reporter is provided for callers who don't care
//! about progress (e.g., CLI batch imports).

/// A single progress event emitted by the import pipeline.
#[derive(Clone, Debug)]
pub enum ImportEvent {
    /// A top-level stage started.
    StageStart {
        /// Short machine-friendly stage name: `"load_mesh"`, `"build_bvh"`,
        /// `"classify_bricks"`, `"voxelize_surface"`, `"smooth_normals"`,
        /// `"write_rkp"`, `"extract_skeleton"`, `"write_rkskel"`.
        stage: &'static str,
        /// Human-readable one-line description for display.
        message: String,
    },
    /// Progress update within a stage. `total == 0` means indeterminate.
    StageProgress {
        /// The stage name from the matching [`StageStart`](Self::StageStart).
        stage: &'static str,
        /// Work units completed so far.
        done: u64,
        /// Total work units, or `0` if unknown.
        total: u64,
    },
    /// A stage completed successfully.
    StageEnd {
        /// The stage name from the matching [`StageStart`](Self::StageStart).
        stage: &'static str,
    },
    /// Non-fatal warning (e.g. missing texture file, malformed UV set).
    Warn {
        /// Free-form warning text.
        message: String,
    },
    /// Fatal error — always paired with an `Err` return from the pipeline.
    Error {
        /// Free-form error text.
        message: String,
    },
}

/// Sink for [`ImportEvent`]s. Implementations might forward to a
/// `mpsc::Sender`, accumulate into a `Vec` for tests, or log to stderr.
///
/// Methods are `&self` so a reporter can be shared across rayon worker
/// threads via `Arc<dyn ProgressReporter>`.
pub trait ProgressReporter: Send + Sync {
    /// Deliver one event. The default implementation discards it.
    fn report(&self, _event: ImportEvent) {}

    /// Polled by the importer between stages (and periodically during
    /// the long parallel voxelize stage). When this returns `true` the
    /// importer aborts with [`crate::ImportError::Cancelled`] at the
    /// next stage boundary. Default implementation always returns
    /// `false` (cancellation disabled); reporters that want to support
    /// cancellation should wrap an `Arc<AtomicBool>` and override.
    fn is_cancelled(&self) -> bool {
        false
    }
}

/// No-op reporter — used as the default when a caller doesn't supply one.
#[derive(Default)]
pub struct NullReporter;

impl ProgressReporter for NullReporter {}

/// Reporter that wraps another reporter + an [`AtomicBool`]-backed
/// cancel flag. The inner reporter sees every event (so a UI still
/// renders progress up to the cancel point); the flag short-circuits
/// the pipeline at the next stage boundary.
///
/// Construct via [`CancelToken::new`] and keep a clone of the handle —
/// calling [`CancelHandle::cancel`] flips the flag.
pub struct CancelToken<R: ProgressReporter> {
    inner: R,
    flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

/// External handle returned from [`CancelToken::new`]. Clone it and
/// hand one to whichever thread wants to fire the cancel.
#[derive(Clone)]
pub struct CancelHandle {
    flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl CancelHandle {
    /// Fire the cancel — the importer will abort at its next
    /// cancellation check.
    pub fn cancel(&self) {
        self.flag.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Check whether cancel has been fired.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl<R: ProgressReporter> CancelToken<R> {
    /// Wrap `inner` with a fresh cancel flag. Returns the wrapping
    /// reporter + a handle the caller can use to fire the cancel
    /// from another thread.
    pub fn new(inner: R) -> (Self, CancelHandle) {
        let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        (
            Self { inner, flag: flag.clone() },
            CancelHandle { flag },
        )
    }
}

impl<R: ProgressReporter> ProgressReporter for CancelToken<R> {
    fn report(&self, event: ImportEvent) {
        self.inner.report(event);
    }
    fn is_cancelled(&self) -> bool {
        self.flag.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Reporter that echoes each event to stderr, matching the old
/// `eprintln!` logging. Useful for CLI and for the editor during
/// migration before the structured-event UI lands.
#[derive(Default)]
pub struct StderrReporter;

impl ProgressReporter for StderrReporter {
    fn report(&self, event: ImportEvent) {
        match event {
            ImportEvent::StageStart { message, .. } => eprintln!("[import] {message}"),
            ImportEvent::StageProgress { stage, done, total } => {
                if total > 0 {
                    eprintln!("[import] {stage}: {done}/{total}");
                } else {
                    eprintln!("[import] {stage}: {done}");
                }
            }
            ImportEvent::StageEnd { stage } => eprintln!("[import] {stage} done"),
            ImportEvent::Warn { message } => eprintln!("[import] WARN: {message}"),
            ImportEvent::Error { message } => eprintln!("[import] ERROR: {message}"),
        }
    }
}
