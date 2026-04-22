//! Command dispatch.
//!
//! `process_command` is the entry point for every `EngineCommand` the
//! editor (or MCP, or a test harness) sends to the engine. The actual
//! arm bodies live in three sibling files — `cmd_edit`, `cmd_scene`,
//! `cmd_runtime` — each handling a chunk of the command enum. This
//! file chains them: each `process_cmd_*` returns `Err(cmd)` when it
//! doesn't match the incoming command, and the next chunk gets a shot.

use crate::command::EngineCommand;

use super::state::EngineState;

impl EngineState {
    pub(crate) fn process_command(&mut self, cmd: EngineCommand) -> bool {
        if matches!(cmd, EngineCommand::Shutdown) {
            return false;
        }
        let cmd = match self.process_cmd_edit(cmd) {
            Ok(()) => return true,
            Err(cmd) => cmd,
        };
        let cmd = match self.process_cmd_scene(cmd) {
            Ok(()) => return true,
            Err(cmd) => cmd,
        };
        match self.process_cmd_runtime(cmd) {
            Ok(()) => {}
            Err(cmd) => {
                eprintln!("[RkpEngine] unhandled command: {cmd:?}");
            }
        }
        true
    }
}
