//! Console log buffer — thread-safe ring buffer for engine log messages.
//!
//! All engine subsystems log via `ConsoleLog::log()`. The engine drains
//! new entries each tick and pushes them to the UI via StateUpdate.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Instant;

const MAX_ENTRIES: usize = 1000;

/// Log severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Info,
    Warn,
    Error,
}

/// A single log entry.
#[derive(Debug, Clone, PartialEq)]
pub struct LogEntry {
    pub level: LogLevel,
    pub message: String,
    /// Seconds since engine start.
    pub timestamp: f32,
}

/// Thread-safe log buffer shared across engine subsystems.
#[derive(Clone)]
pub struct ConsoleLog {
    inner: Arc<Mutex<ConsoleInner>>,
}

struct ConsoleInner {
    entries: VecDeque<LogEntry>,
    /// New entries since last drain (index into entries).
    new_since_drain: usize,
    start_time: Instant,
}

impl ConsoleLog {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(ConsoleInner {
                entries: VecDeque::with_capacity(MAX_ENTRIES),
                new_since_drain: 0,
                start_time: Instant::now(),
            })),
        }
    }

    /// Log a message at the given level.
    pub fn log(&self, level: LogLevel, message: impl Into<String>) {
        let mut inner = self.inner.lock().unwrap();
        let timestamp = inner.start_time.elapsed().as_secs_f32();
        let msg = message.into();

        // Also print to stderr for debugging.
        let prefix = match level {
            LogLevel::Info => "[info]",
            LogLevel::Warn => "[warn]",
            LogLevel::Error => "[ERROR]",
        };
        eprintln!("{prefix} {msg}");

        if inner.entries.len() >= MAX_ENTRIES {
            inner.entries.pop_front();
            if inner.new_since_drain > 0 {
                inner.new_since_drain -= 1;
            }
        }
        inner.entries.push_back(LogEntry {
            level,
            message: msg,
            timestamp,
        });
        inner.new_since_drain += 1;
    }

    /// Convenience methods.
    pub fn info(&self, message: impl Into<String>) {
        self.log(LogLevel::Info, message);
    }

    pub fn warn(&self, message: impl Into<String>) {
        self.log(LogLevel::Warn, message);
    }

    pub fn error(&self, message: impl Into<String>) {
        self.log(LogLevel::Error, message);
    }

    /// Drain new entries since last drain. Returns only the new ones.
    pub fn drain_new(&self) -> Vec<LogEntry> {
        let mut inner = self.inner.lock().unwrap();
        let n = inner.new_since_drain;
        inner.new_since_drain = 0;
        if n == 0 {
            return Vec::new();
        }
        let start = inner.entries.len().saturating_sub(n);
        inner.entries.iter().skip(start).cloned().collect()
    }

    /// Get all entries (for initial load).
    pub fn all_entries(&self) -> Vec<LogEntry> {
        let inner = self.inner.lock().unwrap();
        inner.entries.iter().cloned().collect()
    }

    /// Clear all entries.
    pub fn clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.entries.clear();
        inner.new_since_drain = 0;
    }
}
