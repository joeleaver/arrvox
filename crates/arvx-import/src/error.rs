//! Structured error types for the import pipeline.
//!
//! Variants mirror the pipeline stages from
//! [`crate::voxelize::import_mesh_to_opacity_rkp_with`] so bug
//! reports can point at a specific failure class without parsing
//! free-form error text. The `Display` impl formats the same
//! messages the crate used to return as plain `String`s, so
//! callers that want flat strings can call `.to_string()` at the
//! boundary (see `arvx-convert`, `arvx-engine::import_worker`).

use std::path::PathBuf;

use thiserror::Error;

/// A failure during mesh-to-`.arvx` import.
#[derive(Debug, Error)]
pub enum ImportError {
    /// [`crate::config::ImportConfig::validate`] rejected the config.
    #[error("invalid ImportConfig: {0}")]
    InvalidConfig(String),

    /// Source path has a `.ext` the crate doesn't know how to parse.
    #[error("unsupported mesh format: {0}")]
    UnsupportedFormat(String),

    /// Source file I/O or parse failure (glTF/OBJ/FBX loader error).
    #[error("failed to load mesh '{path}': {reason}")]
    MeshLoad {
        /// Path the loader was attempting.
        path: PathBuf,
        /// Loader-specific error text.
        reason: String,
    },

    /// Skeleton extractor hit unrecoverable input. Different from a
    /// file with *no* skeleton — that's `Ok(None)`, not an error.
    #[error("skeleton extraction failed for '{path}': {reason}")]
    SkeletonExtract {
        /// Path being processed.
        path: PathBuf,
        /// Skeleton-extractor-specific error text.
        reason: String,
    },

    /// Voxelization preconditions failed (e.g. empty mesh).
    #[error("voxelization failed: {0}")]
    Voxelize(String),

    /// Failure during `.arvx` or `.arvxskel` writing (disk full, rename
    /// across filesystems, broken permissions, etc.).
    #[error("write failed: {0}")]
    Write(String),

    /// The caller set the reporter's cancel flag; the importer
    /// aborted at the next stage boundary. Staging files are
    /// cleaned up before this returns.
    #[error("import cancelled")]
    Cancelled,
}

impl ImportError {
    /// Construct an [`ImportError::MeshLoad`] from any error type.
    pub fn mesh_load<E: std::fmt::Display>(path: &std::path::Path, reason: E) -> Self {
        ImportError::MeshLoad {
            path: path.to_path_buf(),
            reason: reason.to_string(),
        }
    }

    /// Construct an [`ImportError::SkeletonExtract`].
    pub fn skeleton<E: std::fmt::Display>(path: &std::path::Path, reason: E) -> Self {
        ImportError::SkeletonExtract {
            path: path.to_path_buf(),
            reason: reason.to_string(),
        }
    }
}
