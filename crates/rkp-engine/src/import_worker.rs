//! Import worker — background thread for mesh-to-.rkp conversion.
//!
//! Accepts import requests via channel, runs `import_mesh_to_opacity_rkp`
//! on a background thread, returns results via another channel.

use std::path::{Path, PathBuf};
use std::sync::mpsc;

use rkf_import::pipeline::{ImportConfig, ImportResult};

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

/// Background import worker.
pub struct ImportWorker {
    request_tx: mpsc::Sender<ImportRequest>,
    result_rx: mpsc::Receiver<ImportCompletion>,
}

impl ImportWorker {
    /// Create and start the import worker thread.
    pub fn new() -> Self {
        let (request_tx, request_rx) = mpsc::channel::<ImportRequest>();
        let (result_tx, result_rx) = mpsc::channel::<ImportCompletion>();

        std::thread::Builder::new()
            .name("rkp-import".into())
            .spawn(move || {
                while let Ok(req) = request_rx.recv() {
                    eprintln!(
                        "[ImportWorker] importing {} → {}",
                        req.source_path.display(),
                        req.output_path.display(),
                    );
                    let result = rkp_render::import_mesh_to_opacity_rkp(
                        &req.source_path,
                        &req.output_path,
                        &req.config,
                    );
                    match &result {
                        Ok(r) => eprintln!(
                            "[ImportWorker] done: {} voxels, {:.1} KB",
                            r.total_bricks,
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
            })
            .expect("failed to spawn import worker");

        Self { request_tx, result_rx }
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
}

/// Build a default ImportConfig for a mesh file.
pub fn default_import_config() -> ImportConfig {
    ImportConfig {
        voxel_size: None, // auto-detect
        lod_levels: 1,
        target_size: 1.0,
        no_normalize: false,
        material_id_override: None,
        import_colors: true,
        rotation_offset: [0.0; 3],
        scale_override: None,
        pool_size: 65536,
        verbose: true,
    }
}

/// Compute the .rkp output path for a source mesh file.
/// e.g., `assets/objects/bunny.glb` → `assets/objects/bunny.rkp`
pub fn rkp_output_path(source: &Path) -> PathBuf {
    source.with_extension("rkp")
}
