//! Import worker — background thread for mesh-to-.arvx conversion.
//!
//! Accepts import requests via channel, runs `import_mesh_to_opacity_rkp_with`
//! on a background thread, and streams structured progress events back to the
//! main thread via a second channel so the UI can render a live status bar.

use std::path::{Path, PathBuf};
use std::sync::mpsc;

use arvx_import::{ImportConfig, ImportEvent, ImportResult, ProgressReporter};

/// An import request.
pub struct ImportRequest {
    pub source_path: PathBuf,
    pub output_path: PathBuf,
    pub config: ImportConfig,
}

/// An import completion.
pub struct ImportCompletion {
    pub source_path: PathBuf,
    pub output_path: PathBuf,
    pub result: Result<ImportResult, String>,
}

/// A progress event tagged with the source path it belongs to, so
/// the engine can multiplex events from multiple concurrent imports.
/// Each import runs on its own dedicated thread (see
/// [`ImportWorker::new`]), so events from different sources interleave
/// freely on `event_rx`; the source_path is how the engine sorts them
/// into the right `ImportProgressInfo` entry.
pub struct TaggedEvent {
    pub source_path: PathBuf,
    pub event: ImportEvent,
}

/// Background import worker.
pub struct ImportWorker {
    request_tx: mpsc::Sender<ImportRequest>,
    result_rx: mpsc::Receiver<ImportCompletion>,
    event_rx: mpsc::Receiver<TaggedEvent>,
}

/// `ProgressReporter` that forwards events onto an `mpsc::Sender`,
/// tagged with the source path the worker is currently processing.
/// Cheap to construct per-import so the worker can create a fresh
/// one each request.
struct MpscReporter {
    source: PathBuf,
    tx: mpsc::Sender<TaggedEvent>,
}

impl ProgressReporter for MpscReporter {
    fn report(&self, event: ImportEvent) {
        // Also echo to stderr so console logs still capture the
        // stage timeline — useful when the UI isn't visible
        // (CLI tools, headless tests). Cheap; stages fire <1 per ms.
        match &event {
            ImportEvent::StageStart { message, .. } => eprintln!("[import] {message}"),
            ImportEvent::Warn { message } => eprintln!("[import] WARN: {message}"),
            ImportEvent::Error { message } => eprintln!("[import] ERROR: {message}"),
            _ => {}
        }
        let _ = self.tx.send(TaggedEvent {
            source_path: self.source.clone(),
            event,
        });
    }
}

impl ImportWorker {
    /// Create and start the import worker.
    ///
    /// Architecture: one long-lived *dispatcher* thread owns
    /// `request_rx` and spawns a fresh per-import worker thread for
    /// every request. Imports therefore run concurrently — clicking
    /// Re-import on two different assets starts two parallel jobs,
    /// each streaming its own events through `event_tx` tagged by
    /// source_path. Internally each import still parallelises via
    /// rayon; the per-import thread just hosts that work and
    /// forwards progress.
    pub fn new() -> Self {
        let (request_tx, request_rx) = mpsc::channel::<ImportRequest>();
        let (result_tx, result_rx) = mpsc::channel::<ImportCompletion>();
        let (event_tx, event_rx) = mpsc::channel::<TaggedEvent>();

        std::thread::Builder::new()
            .name("arvx-import-dispatch".into())
            .spawn(move || {
                while let Ok(req) = request_rx.recv() {
                    let result_tx = result_tx.clone();
                    let event_tx = event_tx.clone();
                    // Per-import thread name helps panics / perf
                    // profiles tell concurrent imports apart. The
                    // thread is detached — we don't need to join it,
                    // its completion arrives on `result_rx`.
                    let thread_name = format!(
                        "arvx-import:{}",
                        req.source_path
                            .file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                            .unwrap_or_else(|| "unknown".into())
                    );
                    if let Err(e) = std::thread::Builder::new()
                        .name(thread_name)
                        .spawn(move || run_import(req, result_tx, event_tx))
                    {
                        eprintln!("[ImportWorker] failed to spawn import thread: {e}");
                    }
                }
            })
            .expect("failed to spawn import dispatcher");

        Self { request_tx, result_rx, event_rx }
    }

    /// Submit an import request (non-blocking).
    pub fn submit(&self, request: ImportRequest) -> bool {
        self.request_tx.send(request).is_ok()
    }

    /// Poll for completed imports (non-blocking).
    pub fn poll_completions(&self) -> Vec<ImportCompletion> {
        let mut completions = Vec::new();
        while let Ok(c) = self.result_rx.try_recv() {
            completions.push(c);
        }
        completions
    }

    /// Poll for queued progress events (non-blocking). Returns all
    /// events that have accumulated since the last call, in FIFO
    /// order. Engine reduces these into per-source progress state.
    pub fn poll_events(&self) -> Vec<TaggedEvent> {
        let mut events = Vec::new();
        while let Ok(e) = self.event_rx.try_recv() {
            events.push(e);
        }
        events
    }
}

/// Run a single import end-to-end. Invoked on a dedicated
/// per-import thread spawned by the dispatcher in [`ImportWorker::new`].
/// All failures (including panics) are reported via `result_tx` so the
/// UI always gets a completion instead of a stuck spinner.
fn run_import(
    req: ImportRequest,
    result_tx: mpsc::Sender<ImportCompletion>,
    event_tx: mpsc::Sender<TaggedEvent>,
) {
    eprintln!(
        "[ImportWorker] importing {} → {}",
        req.source_path.display(),
        req.output_path.display(),
    );
    let source = req.source_path.clone();
    let output = req.output_path.clone();
    let config = req.config.clone();
    let reporter = MpscReporter {
        source: source.clone(),
        tx: event_tx,
    };
    // Panic catch keeps the worker alive across a malformed-input
    // crash so the UI always gets a completion instead of a stuck
    // spinner.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        arvx_import::import_mesh_to_opacity_rkp_with(&source, &output, &config, &reporter)
    }));
    // Flatten the typed `ImportError` to a `String` for
    // `ImportCompletion` — the UI doesn't distinguish error variants,
    // just shows the message. Callers that want structure can import
    // `arvx_import::ImportError` directly.
    let result: Result<ImportResult, String> = match result {
        Ok(Ok(r)) => Ok(r),
        Ok(Err(e)) => Err(e.to_string()),
        Err(payload) => {
            let msg = panic_message(&payload);
            eprintln!("[ImportWorker] panic: {msg}");
            Err(format!("importer panicked: {msg}"))
        }
    };
    match &result {
        Ok(r) => eprintln!(
            "[ImportWorker] done: {} voxels, {:.1} KB",
            r.shell_voxels,
            r.file_size as f64 / 1024.0,
        ),
        Err(e) => eprintln!("[ImportWorker] failed: {e}"),
    }
    let _ = result_tx.send(ImportCompletion {
        source_path: req.source_path,
        output_path: req.output_path,
        result,
    });
}

/// Build a default ImportConfig for a mesh file.
pub fn default_import_config() -> ImportConfig {
    ImportConfig::default()
}

/// Compute the .arvx output path for a source mesh file.
/// e.g., `assets/objects/bunny.glb` → `assets/objects/bunny.arvx`
pub fn arvx_output_path(source: &Path) -> PathBuf {
    source.with_extension("arvx")
}

/// Extract a human-readable message from a `catch_unwind` payload.
/// Panics carry either `&'static str` or `String` in the common cases;
/// fall back to an opaque label for anything exotic.
fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}
