//! Execution phases for the behavior system.

/// Which phase a system runs in.
///
/// The engine executes phases in order: Update → FixedUpdate → (physics) → LateUpdate.
/// Commands are flushed between each phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    /// General gameplay: AI, movement, interactions, game logic.
    /// Receives variable frame delta time.
    Update,
    /// Physics-rate logic. Receives fixed delta time (1/60).
    /// Runs after Update, before the physics step.
    FixedUpdate,
    /// Runs after physics: camera follow, UI sync, cleanup.
    /// Receives variable frame delta time.
    LateUpdate,
}
