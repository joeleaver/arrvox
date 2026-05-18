//! Error and status types for the generator system.
//!
//! `GeneratorError` is what a generator function returns on failure.
//! `GeneratorStatus` tracks the lifecycle of a generator entity for the UI.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Errors returned by generator functions.
#[derive(Debug, Clone)]
pub enum GeneratorError {
    /// Cancel token was set — generator saw it via `ctx.check_cancelled()?`.
    Cancelled,
    /// Author-authored error with a human-readable message.
    Failed(String),
    /// A required param was missing or out of range.
    InvalidParams(String),
}

impl fmt::Display for GeneratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => write!(f, "generation cancelled"),
            Self::Failed(msg) => write!(f, "generation failed: {msg}"),
            Self::InvalidParams(msg) => write!(f, "invalid params: {msg}"),
        }
    }
}

impl std::error::Error for GeneratorError {}

/// Lifecycle status of a generator entity.
///
/// `Pending` → `Stale` → `Generating` → (`Ready` | `Error`).
/// Param edits send `Ready`/`Error` back to `Stale`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GeneratorStatus {
    /// Output is current and matches the params that produced it.
    Ready,
    /// Params changed since the last successful run; output is displayed but stale.
    Stale,
    /// Worker thread is currently running the generator.
    Generating,
    /// Last run failed. The message is the error.
    Error(String),
    /// Newly spawned — no run has completed yet.
    Pending,
}

impl Default for GeneratorStatus {
    fn default() -> Self {
        Self::Pending
    }
}

impl fmt::Display for GeneratorStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ready => write!(f, "Ready"),
            Self::Stale => write!(f, "Stale"),
            Self::Generating => write!(f, "Generating"),
            Self::Error(msg) => write!(f, "Error: {msg}"),
            Self::Pending => write!(f, "Pending"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_status_is_pending() {
        assert_eq!(GeneratorStatus::default(), GeneratorStatus::Pending);
    }

    #[test]
    fn error_display() {
        assert_eq!(GeneratorError::Cancelled.to_string(), "generation cancelled");
        assert_eq!(
            GeneratorError::Failed("oom".into()).to_string(),
            "generation failed: oom",
        );
        assert_eq!(
            GeneratorError::InvalidParams("radius <= 0".into()).to_string(),
            "invalid params: radius <= 0",
        );
    }

    #[test]
    fn status_serde_roundtrip() {
        for s in [
            GeneratorStatus::Ready,
            GeneratorStatus::Stale,
            GeneratorStatus::Generating,
            GeneratorStatus::Error("boom".into()),
            GeneratorStatus::Pending,
        ] {
            let json = serde_json::to_string(&s).unwrap();
            let back: GeneratorStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, s);
        }
    }
}
