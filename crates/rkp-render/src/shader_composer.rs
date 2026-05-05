//! User-shader composition for the deferred shade pass + GPU geometry pass.
//!
//! Scans `<project_root>/assets/shaders/*.wgsl`, parses each shader's
//! optional hook functions, and emits dispatch chunks that get spliced
//! into `rkp_shade.wgsl` (Phase B) and the geometry-build pipeline
//! (Phase C). Both pipelines use the same registry and same
//! `compose()` output structure.
//!
//! ## Module layout (post-split)
//!
//! - [`types`] — public data types only (`ParamDef`, `ShaderMetadata`,
//!   `UserShaderEntry`, `UserShaderRegistry`, `UserShaderInfo`,
//!   `ShaderComposerError`, `ComposedChunks`).
//! - [`parser`] — `scan_dir` + `parse_file` + low-level scanner.
//! - [`compose`] — `compose` + per-pipeline chunk emitters +
//!   `splice_inst_chunks`.
//! - [`hash`] — `fnv1a_64` + the registry hash used for cache keys.
//!
//! All previously-public symbols are re-exported at this level so
//! `rkp_render::shader_composer::Foo` keeps working unchanged.
//!
//! ## Authoring contract
//!
//! Each `*.wgsl` file is one shader, named by its file stem
//! (`assets/shaders/grass.wgsl` → "grass"). A shader provides up to
//! four hooks; the function name signals which hook:
//!
//! ```ignore
//! fn user_grass_pre(world_pos: vec3<f32>, ctx: UserCtx) -> vec3<f32>
//! fn user_grass_generate(world_pos: vec3<f32>, ctx: UserCtx) -> TreeSample
//! fn user_grass_post(child: TreeSample, world_pos: vec3<f32>, ctx: UserCtx) -> TreeSample
//! fn user_grass_envelope(ctx: UserCtx) -> f32
//! ```
//!
//! Hooks not present default to identity (`pre` returns `world_pos`,
//! `post` returns `child`, `envelope` returns `0`, `generate` returns
//! a miss). Files that declare no hooks are still legal — they're
//! registered but contribute no behavior.
//!
//! ## Composition strategy
//!
//! 1. Each user function is captured verbatim from the source file
//!    (full `fn ... { ... }` text, brace-matched).
//! 2. The function name `user_<name>_<hook>` is rewritten to
//!    `rkp_user_<id>_<hook>` so dispatch can call it by a stable name
//!    independent of the user's choice of `<name>`.
//! 3. Four `dispatch_user_*` switches are emitted, one per hook. Each
//!    switch routes by `shader_id` to the matching `rkp_user_<id>_<hook>`
//!    function; shaders that don't provide that hook fall through to
//!    the switch's default (identity).
//!
//! `shader_id` 0 is reserved for "no shader" — the default arms
//! return the identity behavior. Registered shaders get ids 1..=N in
//! filesystem-walk order.

pub mod compose;
pub mod hash;
pub mod lib_symbols;
pub mod parser;
pub mod types;

// Public re-exports — keep `rkp_render::shader_composer::Foo` stable.
pub use compose::{compose, splice_const_marker, splice_emit_chunks, splice_inst_chunks};
pub use hash::fnv1a_64;
pub use parser::{parse_file, scan_dir};
pub use types::{
    ComposedChunks, ParamDef, ShaderComposerError, ShaderMetadata, UserShaderEntry,
    UserShaderInfo, UserShaderRegistry,
};

#[cfg(test)]
mod tests;
