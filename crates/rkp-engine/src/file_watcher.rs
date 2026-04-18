//! File watcher for RKIPatch.
//!
//! Watches project assets/ recursively. Classifies events by extension.
//! Polls non-blocking each tick.

use std::path::{Path, PathBuf};
use std::sync::mpsc;

use notify::{RecommendedWatcher, RecursiveMode, Watcher};

/// File change event types relevant to RKIPatch.
#[derive(Debug, Clone)]
pub enum FileEvent {
    /// A .rkp model file was created or modified.
    ModelChanged(PathBuf),
    /// A .wgsl shader file was created or modified.
    ShaderChanged(PathBuf),
    /// A .rkmat material file was created or modified.
    MaterialChanged(PathBuf),
    /// An importable mesh (.glb, .gltf, .obj, .fbx) was created or modified.
    MeshSourceChanged(PathBuf),
    /// A .rs script file was created or modified.
    ScriptChanged(PathBuf),
}

/// Watches project directories for asset changes.
pub struct RkpFileWatcher {
    rx: mpsc::Receiver<FileEvent>,
    _watcher: RecommendedWatcher,
}

impl RkpFileWatcher {
    /// Start watching the given directories recursively.
    pub fn new(watch_paths: &[&Path]) -> Result<Self, String> {
        let (tx, rx) = mpsc::channel();

        let event_tx = tx;
        let mut watcher = notify::recommended_watcher(move |res: Result<notify::Event, notify::Error>| {
            let Ok(event) = res else { return };

            use notify::EventKind;
            match event.kind {
                EventKind::Create(_) | EventKind::Modify(_) => {}
                _ => return,
            }

            for path in &event.paths {
                let ext = path.extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("");
                let file_event = match ext {
                    "rkp" => Some(FileEvent::ModelChanged(path.clone())),
                    "wgsl" => Some(FileEvent::ShaderChanged(path.clone())),
                    "rkmat" => Some(FileEvent::MaterialChanged(path.clone())),
                    "glb" | "gltf" | "obj" | "fbx" => Some(FileEvent::MeshSourceChanged(path.clone())),
                    "rs" => Some(FileEvent::ScriptChanged(path.clone())),
                    _ => None,
                };
                if let Some(fe) = file_event {
                    let _ = event_tx.send(fe);
                }
            }
        }).map_err(|e| format!("create watcher: {e}"))?;

        for path in watch_paths {
            if path.exists() {
                watcher.watch(path, RecursiveMode::Recursive)
                    .map_err(|e| format!("watch {}: {e}", path.display()))?;
            }
        }

        Ok(Self { rx, _watcher: watcher })
    }

    /// Poll for pending file events (non-blocking, deduplicated).
    pub fn poll_events(&self) -> Vec<FileEvent> {
        let mut events = Vec::new();
        while let Ok(event) = self.rx.try_recv() {
            let dominated = events.iter().any(|existing: &FileEvent| match (existing, &event) {
                (FileEvent::ModelChanged(a), FileEvent::ModelChanged(b)) => a == b,
                (FileEvent::ShaderChanged(a), FileEvent::ShaderChanged(b)) => a == b,
                (FileEvent::MaterialChanged(a), FileEvent::MaterialChanged(b)) => a == b,
                (FileEvent::MeshSourceChanged(a), FileEvent::MeshSourceChanged(b)) => a == b,
                (FileEvent::ScriptChanged(a), FileEvent::ScriptChanged(b)) => a == b,
                _ => false,
            });
            if !dominated {
                events.push(event);
            }
        }
        events
    }
}
